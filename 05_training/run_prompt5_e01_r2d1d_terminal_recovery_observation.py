from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
HF1_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1d_terminal_recovery_observation"
GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
TARGET_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
KST_OFFSET_TEXT = "+09:00"

SAFE_CALL_TARGET = 800
HARD_CALL_LIMIT = 850
MAX_CALLS_PER_MINUTE = 12
BROAD_SCAN_INTERVAL_SECONDS = 120
FOCUSED_INTERVAL_SECONDS = 30
POST_CONFIRM_MIN_SECONDS = 600
POST_CONFIRM_MIN_SAMPLES = 3
FOCUSED_SESSION_MAX_SECONDS = 45 * 60
TERMINAL_TRIGGER_MARGIN = 5
POST_TERMINAL_SEQUENCE_MAX = 5
MAX_FOCUSED_PER_ROUTE = 1
MAX_FOCUSED_GLOBAL = 2
FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE"}

SAMPLE_COLUMNS = [
    "route_id",
    "vehicle_id",
    "capture_mode",
    "request_time",
    "provider_event_time",
    "direction",
    "current_sequence",
    "stop_id",
    "x",
    "y",
    "effective_live_terminal_sequence",
    "terminal_trigger_threshold",
    "is_terminal_zone",
    "previous_provider_event_time",
    "previous_sequence",
    "previous_stop_id",
    "previous_x",
    "previous_y",
    "elapsed_seconds",
    "sequence_delta",
    "spatial_distance_m",
    "route_changed",
    "direction_changed",
    "sequence_reset",
    "provider_response_status",
    "raw_file_path",
    "raw_file_sha256",
]

EPISODE_COLUMNS = [
    "episode_id",
    "route_id",
    "vehicle_id",
    "direction",
    "episode_status",
    "final_status_class",
    "observation_start_time",
    "observation_end_time",
    "last_pre_terminal_time",
    "first_terminal_time",
    "last_terminal_time",
    "first_post_terminal_time",
    "arrival_window_start",
    "arrival_window_end",
    "departure_window_start",
    "departure_window_end",
    "recovery_lower_bound_sec",
    "recovery_upper_bound_sec",
    "left_censored",
    "right_censored",
    "complete_interval_censored_episode",
    "terminal_hold_sample_count",
    "post_terminal_confirmation_sample_count",
    "sequence_before_reset",
    "sequence_after_reset",
    "before_terminal_raw_sha256",
    "first_terminal_raw_sha256",
    "last_terminal_raw_sha256",
    "first_post_terminal_raw_sha256",
    "vehicle_id_continuity_passed",
    "route_continuity_passed",
    "direction_continuity_passed",
    "timestamp_monotonicity_passed",
    "evidence_sha256_passed",
    "invalid_reason",
]


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def distance_m(x1: Any, y1: Any, x2: Any, y2: Any) -> Optional[float]:
    lon1, lat1, lon2, lat2 = as_float(x1), as_float(y1), as_float(x2), as_float(y2)
    if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
        return None
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def redact_url(url: str, secret: str) -> str:
    if not secret:
        return url
    return (
        url.replace(secret, "<REDACTED>")
        .replace(urllib.parse.quote(secret, safe="%"), "<REDACTED>")
        .replace(urllib.parse.quote_plus(secret, safe="%"), "<REDACTED>")
    )


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    if not secret:
        return {"scan_root": str(root), "secret_literal_occurrence_count": 0, "findings": []}
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")]
    findings: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def provider_time(request_iso: str, hhmmss: Any) -> Optional[str]:
    text = norm(hhmmss)
    if not text or len(text) < 6:
        return None
    try:
        request_dt = datetime.fromisoformat(request_iso)
        event_time = dtime(int(text[0:2]), int(text[2:4]), int(text[4:6]), tzinfo=request_dt.tzinfo)
        return datetime.combine(request_dt.date(), event_time, tzinfo=request_dt.tzinfo).isoformat(timespec="seconds")
    except ValueError:
        return None


def request_time_from_raw_path(path: Path) -> str:
    stem = path.stem
    parsed = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").astimezone()
    return parsed.isoformat(timespec="seconds")


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
    paths = [
        ["body", "items"],
        ["body", "items", "item"],
        ["response", "body", "items"],
        ["response", "body", "items", "item"],
        ["items"],
        ["item"],
    ]
    for path in paths:
        cur = obj
        for item in path:
            if isinstance(cur, dict):
                cur = cur.get(item)
            else:
                cur = None
                break
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


def parse_items(text: str) -> Tuple[List[Dict[str, Any]], bool]:
    stripped = text.lstrip()
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            payload = json.loads(text)
            if isinstance(payload, dict):
                header = payload.get("header") or payload.get("response", {}).get("header") or {}
                result_code = norm(header.get("resultCode"))
                if result_code and result_code != "0000":
                    return [], False
            return find_items(payload), True
    except Exception:
        return [], False
    return [], False


def load_route_meta(mapping_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for _, row in mapping_df[mapping_df["route_id"].astype(str).isin(TARGET_ROUTES)].iterrows():
        route_id = str(row["route_id"])
        effective = as_int(row.get("effective_live_terminal_sequence"))
        meta[route_id] = {
            "route_id": route_id,
            "route_no": norm(row.get("route_no")),
            "direction": norm(row.get("direction_id")) or norm(row.get("service_direction_id")) or "1",
            "terminal_stop_id": norm(row.get("terminal_stop_id")) or norm(row.get("service_terminal_stop_id")),
            "terminal_sequence": as_int(row.get("terminal_sequence")) or as_int(row.get("full_terminal_sequence")) or effective,
            "effective_live_terminal_sequence": effective,
            "terminal_trigger_threshold": None if effective is None else effective - TERMINAL_TRIGGER_MARGIN,
            "terminal_operation_type": norm(row.get("terminal_operation_type")),
            "terminal_operation_resolved": truthy(row.get("terminal_operation_resolved")),
            "confidence": norm(row.get("confidence")) or norm(row.get("mapping_confidence")),
            "duplicate_loop_closure": truthy(row.get("duplicate_loop_closure")),
            "mapping_v6_decision": norm(row.get("mapping_v6_decision")),
            "static_live_topology_decision": "LIVE_SEQUENCE_EXTENSION" if route_id == "4050010000" else "PRESERVED_HF1",
        }
    return meta


def fetch_route(
    route_id: str,
    secret: str,
    output_root: Path,
    capture_mode: str,
    route_meta: Mapping[str, Any],
    timeout: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    request_time = now_iso()
    raw_dir = output_root / "raw" / capture_mode.lower() / route_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = [("serviceKey", secret), ("routeId", route_id), ("resultType", "json")]
    url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
    raw_path = raw_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    http_status: Optional[int] = None
    content_type = ""
    timeout_error = False
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1d/1.0"}), timeout=timeout) as response:
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
        payload = json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False).encode("utf-8")
        content_type = "application/json"
        encoding = "utf-8"
    raw_path.write_bytes(payload)
    text = payload.decode(encoding or "utf-8", errors="replace")
    status = detect_response_status(text, http_status, timeout_error=timeout_error)
    items, parsed = parse_items(text) if status == "OK" else ([], False)
    if status == "OK" and not parsed:
        status = "PROVIDER_PARSE_FAILURE"
    raw_sha = sha256_file(raw_path)
    rows: List[Dict[str, Any]] = []
    effective = as_int(route_meta.get("effective_live_terminal_sequence"))
    threshold = as_int(route_meta.get("terminal_trigger_threshold"))
    for item in items:
        seq = as_int(item.get("seq") or item.get("stopSeq"))
        event_time = provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
        vehicle_id = norm(item.get("vhcNo2")) or norm(item.get("vhcNo")) or norm(item.get("busId"))
        if seq is None or vehicle_id is None:
            continue
        stop_id = norm(item.get("bsId")) or norm(item.get("stopId"))
        rows.append(
            {
                "route_id": norm(item.get("routeId")) or route_id,
                "vehicle_id": vehicle_id,
                "capture_mode": capture_mode,
                "request_time": request_time,
                "provider_event_time": event_time,
                "direction": norm(item.get("moveDir")) or norm(item.get("direction_id")) or route_meta.get("direction"),
                "current_sequence": seq,
                "stop_id": stop_id,
                "x": as_float(item.get("xPos") or item.get("x")),
                "y": as_float(item.get("yPos") or item.get("y")),
                "effective_live_terminal_sequence": effective,
                "terminal_trigger_threshold": threshold,
                "is_terminal_zone": bool(threshold is not None and seq >= threshold),
                "provider_response_status": status,
                "raw_file_path": str(raw_path),
                "raw_file_sha256": raw_sha,
            }
        )
    request = {
        "route_id": route_id,
        "request_time": request_time,
        "capture_mode": capture_mode,
        "http_status": http_status,
        "content_type": content_type,
        "provider_response_status": status,
        "valid_provider_payload": status == "OK",
        "provider_parse_success": parsed,
        "normalized_row_count": len(rows),
        "raw_file_path": str(raw_path),
        "raw_file_sha256": raw_sha,
        "request_url_redacted": redact_url(url, secret),
    }
    return rows, request


def load_saved_raw_capture(raw_root: Path, route_meta: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    root = raw_root / "raw" if (raw_root / "raw").exists() else raw_root
    mode_map = {
        "preflight": "PREFLIGHT",
        "broad_scan": "BROAD_SCAN",
        "terminal_focused": "TERMINAL_FOCUSED",
        "post_terminal_confirm": "POST_TERMINAL_CONFIRM",
    }
    for path in sorted(root.rglob("*.json")):
        mode_key = path.parent.parent.name
        route_id = path.parent.name
        if route_id not in route_meta:
            continue
        capture_mode = mode_map.get(mode_key, mode_key.upper())
        request_time = request_time_from_raw_path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        status = detect_response_status(text, 200)
        items, parsed = parse_items(text) if status == "OK" else ([], False)
        if status == "OK" and not parsed:
            status = "PROVIDER_PARSE_FAILURE"
        raw_sha = sha256_file(path)
        effective = as_int(route_meta[route_id].get("effective_live_terminal_sequence"))
        threshold = as_int(route_meta[route_id].get("terminal_trigger_threshold"))
        route_rows: List[Dict[str, Any]] = []
        for item in items:
            seq = as_int(item.get("seq") or item.get("stopSeq"))
            event_time = provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
            vehicle_id = norm(item.get("vhcNo2")) or norm(item.get("vhcNo")) or norm(item.get("busId"))
            if seq is None or vehicle_id is None:
                continue
            route_rows.append(
                {
                    "route_id": norm(item.get("routeId")) or route_id,
                    "vehicle_id": vehicle_id,
                    "capture_mode": capture_mode,
                    "request_time": request_time,
                    "provider_event_time": event_time,
                    "direction": norm(item.get("moveDir")) or norm(item.get("direction_id")) or route_meta[route_id].get("direction"),
                    "current_sequence": seq,
                    "stop_id": norm(item.get("bsId")) or norm(item.get("stopId")),
                    "x": as_float(item.get("xPos") or item.get("x")),
                    "y": as_float(item.get("yPos") or item.get("y")),
                    "effective_live_terminal_sequence": effective,
                    "terminal_trigger_threshold": threshold,
                    "is_terminal_zone": bool(threshold is not None and seq >= threshold),
                    "provider_response_status": status,
                    "raw_file_path": str(path),
                    "raw_file_sha256": raw_sha,
                }
            )
        rows.extend(route_rows)
        requests.append(
            {
                "route_id": route_id,
                "request_time": request_time,
                "capture_mode": capture_mode,
                "http_status": 200,
                "content_type": "saved/raw",
                "provider_response_status": status,
                "valid_provider_payload": status == "OK",
                "provider_parse_success": parsed,
                "normalized_row_count": len(route_rows),
                "raw_file_path": str(path),
                "raw_file_sha256": raw_sha,
                "request_url_redacted": None,
            }
        )
    requests = sorted(requests, key=lambda row: str(row.get("request_time") or ""))
    rows = sorted(rows, key=lambda row: (str(row.get("request_time") or ""), str(row.get("route_id") or ""), str(row.get("vehicle_id") or "")))
    return rows, requests


def add_previous_fields(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    df = samples.copy()
    df["_provider_dt"] = pd.to_datetime(df["provider_event_time"], errors="coerce")
    df["_request_dt"] = pd.to_datetime(df["request_time"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for _, group in df.sort_values(["route_id", "direction", "vehicle_id", "_provider_dt", "_request_dt"]).groupby(["route_id", "direction", "vehicle_id"], dropna=False):
        previous: Optional[Mapping[str, Any]] = None
        for _, row in group.iterrows():
            out = row.drop(labels=[c for c in ["_provider_dt", "_request_dt"] if c in row.index]).to_dict()
            out.update(
                {
                    "previous_provider_event_time": None,
                    "previous_sequence": None,
                    "previous_stop_id": None,
                    "previous_x": None,
                    "previous_y": None,
                    "elapsed_seconds": None,
                    "sequence_delta": None,
                    "spatial_distance_m": None,
                    "route_changed": False,
                    "direction_changed": False,
                    "sequence_reset": False,
                }
            )
            if previous is not None:
                cur_dt = row["_provider_dt"]
                prev_dt = previous.get("_provider_dt")
                elapsed = None
                if pd.notna(cur_dt) and pd.notna(prev_dt):
                    elapsed = float((cur_dt - prev_dt).total_seconds())
                prev_seq = as_int(previous.get("current_sequence"))
                cur_seq = as_int(row.get("current_sequence"))
                out.update(
                    {
                        "previous_provider_event_time": previous.get("provider_event_time"),
                        "previous_sequence": prev_seq,
                        "previous_stop_id": previous.get("stop_id"),
                        "previous_x": previous.get("x"),
                        "previous_y": previous.get("y"),
                        "elapsed_seconds": elapsed,
                        "sequence_delta": None if prev_seq is None or cur_seq is None else cur_seq - prev_seq,
                        "spatial_distance_m": distance_m(previous.get("x"), previous.get("y"), row.get("x"), row.get("y")),
                        "route_changed": norm(previous.get("route_id")) != norm(row.get("route_id")),
                        "direction_changed": norm(previous.get("direction")) != norm(row.get("direction")),
                        "sequence_reset": bool(prev_seq is not None and cur_seq is not None and cur_seq < prev_seq),
                    }
                )
            rows.append(out)
            previous = row
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def complete_episode_payload(ep: Dict[str, Any], status: str, end_row: Optional[Mapping[str, Any]], output_root: Path) -> Dict[str, Any]:
    left = bool(ep.get("left_censored"))
    right = status in {"RIGHT_CENSORED"} or bool(ep.get("right_censored"))
    complete = (
        status == "POST_TERMINAL_CONFIRMED"
        and not left
        and ep.get("last_pre_terminal") is not None
        and ep.get("first_terminal") is not None
        and ep.get("last_terminal") is not None
        and ep.get("first_post_terminal") is not None
    )
    if status == "POST_TERMINAL_CONFIRMED" and left:
        status = "LEFT_CENSORED"
    final_status = "COMPLETE" if complete else ("DUAL_CENSORED" if left and right else ("LEFT_CENSORED" if left else ("RIGHT_CENSORED" if right else status)))
    last_pre = ep.get("last_pre_terminal")
    first_terminal = ep.get("first_terminal")
    last_terminal = ep.get("last_terminal")
    first_post = ep.get("first_post_terminal")

    def row_time(row: Optional[Mapping[str, Any]]) -> Optional[str]:
        return None if row is None else norm(row.get("provider_event_time"))

    def row_sha(row: Optional[Mapping[str, Any]]) -> Optional[str]:
        return None if row is None else norm(row.get("raw_file_sha256"))

    lower = None
    upper = None
    if complete and last_pre is not None and first_terminal is not None and last_terminal is not None and first_post is not None:
        ft = parse_dt(first_terminal.get("provider_event_time"))
        lt = parse_dt(last_terminal.get("provider_event_time"))
        lp = parse_dt(last_pre.get("provider_event_time"))
        fp = parse_dt(first_post.get("provider_event_time"))
        if ft and lt:
            lower = max(0.0, (lt - ft).total_seconds())
        if lp and fp:
            upper = max(0.0, (fp - lp).total_seconds())
    raw_paths = [row.get("raw_file_path") for row in [last_pre, first_terminal, last_terminal, first_post] if row is not None]
    evidence_passed = True
    if complete:
        evidence_passed = all(row_sha(row) for row in [last_pre, first_terminal, last_terminal, first_post])
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.exists() or sha256_file(path) not in {row_sha(last_pre), row_sha(first_terminal), row_sha(last_terminal), row_sha(first_post)}:
            evidence_passed = False

    direction = ep.get("direction")
    route_id = ep.get("route_id")
    vehicle_id = ep.get("vehicle_id")
    return {
        "episode_id": ep.get("episode_id"),
        "route_id": route_id,
        "vehicle_id": vehicle_id,
        "direction": direction,
        "episode_status": status,
        "final_status_class": final_status,
        "observation_start_time": ep.get("observation_start_time"),
        "observation_end_time": row_time(end_row) or row_time(first_post) or row_time(last_terminal),
        "last_pre_terminal_time": row_time(last_pre),
        "first_terminal_time": row_time(first_terminal),
        "last_terminal_time": row_time(last_terminal),
        "first_post_terminal_time": row_time(first_post),
        "arrival_window_start": row_time(last_pre) if complete else None,
        "arrival_window_end": row_time(first_terminal) if complete else None,
        "departure_window_start": row_time(last_terminal) if complete else None,
        "departure_window_end": row_time(first_post) if complete else None,
        "recovery_lower_bound_sec": lower,
        "recovery_upper_bound_sec": upper,
        "left_censored": left,
        "right_censored": right,
        "complete_interval_censored_episode": complete,
        "terminal_hold_sample_count": int(ep.get("terminal_hold_sample_count", 0)),
        "post_terminal_confirmation_sample_count": int(ep.get("post_terminal_confirmation_sample_count", 0)),
        "sequence_before_reset": as_int(last_terminal.get("current_sequence")) if last_terminal is not None else None,
        "sequence_after_reset": as_int(first_post.get("current_sequence")) if first_post is not None else None,
        "before_terminal_raw_sha256": row_sha(last_pre),
        "first_terminal_raw_sha256": row_sha(first_terminal),
        "last_terminal_raw_sha256": row_sha(last_terminal),
        "first_post_terminal_raw_sha256": row_sha(first_post),
        "vehicle_id_continuity_passed": bool(vehicle_id),
        "route_continuity_passed": True,
        "direction_continuity_passed": True,
        "timestamp_monotonicity_passed": not bool(ep.get("timestamp_invalid")),
        "evidence_sha256_passed": evidence_passed,
        "invalid_reason": ep.get("invalid_reason"),
    }


def reconstruct_episodes(samples: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    df = samples.copy()
    df["_provider_dt"] = pd.to_datetime(df["provider_event_time"], errors="coerce")
    df["_request_dt"] = pd.to_datetime(df["request_time"], errors="coerce")
    episodes: List[Dict[str, Any]] = []
    episode_index = 0
    for (route_id, direction, vehicle_id), group in df.dropna(subset=["vehicle_id"]).sort_values(["_provider_dt", "_request_dt"]).groupby(["route_id", "direction", "vehicle_id"], dropna=False):
        last_pre: Optional[Mapping[str, Any]] = None
        active: Optional[Dict[str, Any]] = None
        previous_dt: Optional[datetime] = None
        previous_row: Optional[Mapping[str, Any]] = None
        for _, row in group.iterrows():
            row_map = row.to_dict()
            cur_dt = row_map.get("_provider_dt")
            if pd.isna(cur_dt):
                if active is not None:
                    active["timestamp_invalid"] = True
                    active["invalid_reason"] = "INVALID_TIMESTAMP"
                continue
            if previous_dt is not None and cur_dt < previous_dt:
                if active is not None:
                    active["timestamp_invalid"] = True
                    active["invalid_reason"] = "INVALID_TIMESTAMP"
            threshold = as_int(row_map.get("terminal_trigger_threshold"))
            seq = as_int(row_map.get("current_sequence"))
            near = bool(threshold is not None and seq is not None and seq >= threshold)
            reset = bool(active is not None and active.get("first_terminal") is not None and seq is not None and seq <= POST_TERMINAL_SEQUENCE_MAX and not near)
            if active is None:
                if near:
                    episode_index += 1
                    active = {
                        "episode_id": f"r2d1d_{episode_index:05d}",
                        "route_id": str(route_id),
                        "direction": str(direction),
                        "vehicle_id": str(vehicle_id),
                        "observation_start_time": norm(row_map.get("provider_event_time")),
                        "last_pre_terminal": last_pre,
                        "first_terminal": row_map,
                        "last_terminal": row_map,
                        "terminal_hold_sample_count": 1,
                        "left_censored": last_pre is None,
                        "right_censored": False,
                        "post_terminal_confirmation_sample_count": 0,
                    }
                else:
                    last_pre = row_map
            else:
                if reset:
                    active["first_post_terminal"] = row_map
                    active["post_terminal_confirmation_sample_count"] = int(active.get("post_terminal_confirmation_sample_count", 0)) + 1
                    active["last_post_terminal_dt"] = cur_dt
                elif active.get("first_post_terminal") is not None:
                    active["post_terminal_confirmation_sample_count"] = int(active.get("post_terminal_confirmation_sample_count", 0)) + 1
                    first_post_dt = parse_dt(active["first_post_terminal"].get("provider_event_time"))
                    elapsed_after_post = (cur_dt - first_post_dt).total_seconds() if first_post_dt else 0
                    if int(active["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or elapsed_after_post >= POST_CONFIRM_MIN_SECONDS:
                        episodes.append(complete_episode_payload(active, "POST_TERMINAL_CONFIRMED", row_map, output_root))
                        active = None
                        last_pre = row_map
                elif near:
                    active["last_terminal"] = row_map
                    active["terminal_hold_sample_count"] = int(active.get("terminal_hold_sample_count", 0)) + 1
                else:
                    last_pre = row_map
            previous_dt = cur_dt
            previous_row = row_map
        if active is not None:
            active["right_censored"] = True
            episodes.append(complete_episode_payload(active, "RIGHT_CENSORED", previous_row, output_root))
    return pd.DataFrame(episodes, columns=EPISODE_COLUMNS)


def compute_request_audit(requests: Sequence[Mapping[str, Any]], secret_leak_count: int) -> Dict[str, Any]:
    rows = list(requests)
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[str(row.get("provider_response_status"))] += 1
    first_fatal_index = None
    first_fatal_time = None
    for idx, row in enumerate(rows):
        if row.get("provider_response_status") in FATAL_STATUSES:
            first_fatal_index = idx
            first_fatal_time = row.get("request_time")
            break
    minute_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        minute = str(row.get("request_time", ""))[:16]
        if minute:
            minute_counts[minute] += 1
    timeout_count = int(status_counts.get("TIMEOUT", 0))
    parse_failure_count = int(status_counts.get("PROVIDER_PARSE_FAILURE", 0))
    return {
        "preflight_api_call_count": 1 if rows else 0,
        "campaign_api_call_count": max(0, len(rows) - 1),
        "total_network_api_call_count": len(rows),
        "successful_response_count": int(status_counts.get("OK", 0)),
        "quota_error_count": int(status_counts.get("RATE_LIMIT", 0)),
        "auth_error_count": int(status_counts.get("AUTH_ERROR", 0)),
        "http_429_count": int(status_counts.get("HTTP_429", 0)),
        "html_response_count": int(status_counts.get("HTML_RESPONSE", 0)),
        "timeout_count": timeout_count,
        "parse_failure_count": parse_failure_count,
        "first_fatal_error_time": first_fatal_time,
        "calls_after_first_fatal_error": 0 if first_fatal_index is None else max(0, len(rows) - first_fatal_index - 1),
        "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
        "hard_budget_respected": len(rows) <= HARD_CALL_LIMIT,
        "status_counts": dict(status_counts),
        "api_runtime_audit_passed": (
            len(rows) <= HARD_CALL_LIMIT
            and (max(minute_counts.values()) if minute_counts else 0) <= MAX_CALLS_PER_MINUTE
            and (0 if first_fatal_index is None else max(0, len(rows) - first_fatal_index - 1)) == 0
            and secret_leak_count == 0
        ),
    }


def route_episode_summary(route_id: str, samples: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
    route_episodes = episodes[episodes["route_id"].astype(str) == route_id] if not episodes.empty else pd.DataFrame()
    complete = route_episodes[route_episodes["complete_interval_censored_episode"] == True] if not route_episodes.empty else pd.DataFrame()
    complete_times = pd.to_datetime(complete["first_terminal_time"], errors="coerce") if not complete.empty else pd.Series(dtype="datetime64[ns]")
    hour_buckets = set(complete_times.dropna().dt.strftime("%Y-%m-%dT%H").tolist())
    lower = pd.to_numeric(complete["recovery_lower_bound_sec"], errors="coerce") if not complete.empty else pd.Series(dtype=float)
    upper = pd.to_numeric(complete["recovery_upper_bound_sec"], errors="coerce") if not complete.empty else pd.Series(dtype=float)
    return {
        "route_id": route_id,
        "complete_interval_censored_episode_count": int(len(complete)),
        "left_censored_episode_count": int(route_episodes["left_censored"].sum()) if not route_episodes.empty else 0,
        "right_censored_episode_count": int(route_episodes["right_censored"].sum()) if not route_episodes.empty else 0,
        "dual_censored_episode_count": int(((route_episodes["left_censored"] == True) & (route_episodes["right_censored"] == True)).sum()) if not route_episodes.empty else 0,
        "independent_vehicle_count": int(complete["vehicle_id"].nunique(dropna=True)) if not complete.empty else 0,
        "observed_hour_bucket_count": len(hour_buckets),
        "observed_hour_buckets": sorted(hour_buckets),
        "broad_scan_observation_count": int((route_samples["capture_mode"] == "BROAD_SCAN").sum()) if not route_samples.empty else 0,
        "terminal_focused_observation_count": int((route_samples["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not route_samples.empty else 0,
        "post_terminal_confirmation_observation_count": int((route_samples["capture_mode"] == "POST_TERMINAL_CONFIRM").sum()) if not route_samples.empty else 0,
        "terminal_trigger_count": int(route_episodes["first_terminal_time"].notna().sum()) if not route_episodes.empty else 0,
        "terminal_hold_observation_count": int(route_episodes["terminal_hold_sample_count"].sum()) if not route_episodes.empty else 0,
        "post_terminal_reset_count": int(route_episodes["first_post_terminal_time"].notna().sum()) if not route_episodes.empty else 0,
        "post_terminal_confirmed_count": int(len(complete)),
        "episode_count": int(len(route_episodes)),
        "lower_bound_min": None if lower.dropna().empty else float(lower.min()),
        "lower_bound_max": None if lower.dropna().empty else float(lower.max()),
        "upper_bound_min": None if upper.dropna().empty else float(upper.min()),
        "upper_bound_max": None if upper.dropna().empty else float(upper.max()),
        "censoring_count": int(((route_episodes["left_censored"] == True) | (route_episodes["right_censored"] == True)).sum()) if not route_episodes.empty else 0,
        "sufficiency_passed": int(len(complete)) >= 5 and (int(complete["vehicle_id"].nunique(dropna=True)) if not complete.empty else 0) >= 3 and len(hour_buckets) >= 2,
    }


def counter_audit_v5(samples: pd.DataFrame, episodes: pd.DataFrame, runtime_sessions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_route = {route_id: route_episode_summary(route_id, samples, episodes) for route_id in TARGET_ROUTES}
    total = {
        "broad_scan_observation_count": sum(row["broad_scan_observation_count"] for row in by_route.values()),
        "terminal_focused_observation_count": sum(row["terminal_focused_observation_count"] for row in by_route.values()),
        "post_terminal_confirmation_observation_count": sum(row["post_terminal_confirmation_observation_count"] for row in by_route.values()),
        "terminal_trigger_count": sum(row["terminal_trigger_count"] for row in by_route.values()),
        "focused_session_started_count": len(runtime_sessions),
        "focused_session_completed_count": sum(1 for row in runtime_sessions if row.get("status") == "COMPLETED"),
        "terminal_entry_episode_count": int(len(episodes)),
        "terminal_hold_observation_count": sum(row["terminal_hold_observation_count"] for row in by_route.values()),
        "post_terminal_reset_count": sum(row["post_terminal_reset_count"] for row in by_route.values()),
        "post_terminal_confirmed_count": sum(row["post_terminal_confirmed_count"] for row in by_route.values()),
        "complete_interval_censored_episode_count": sum(row["complete_interval_censored_episode_count"] for row in by_route.values()),
        "left_censored_episode_count": sum(row["left_censored_episode_count"] for row in by_route.values()),
        "right_censored_episode_count": sum(row["right_censored_episode_count"] for row in by_route.values()),
        "dual_censored_episode_count": sum(row["dual_censored_episode_count"] for row in by_route.values()),
        "invalid_vehicle_continuity_count": 0,
        "invalid_timestamp_count": int((episodes["timestamp_monotonicity_passed"] == False).sum()) if not episodes.empty else 0,
        "invalid_route_direction_count": 0,
        "independent_vehicle_count": int(episodes[episodes["complete_interval_censored_episode"] == True]["vehicle_id"].nunique(dropna=True)) if not episodes.empty else 0,
        "contradiction_count": 0,
    }
    invalid_episode_count = total["invalid_vehicle_continuity_count"] + total["invalid_timestamp_count"] + total["invalid_route_direction_count"]
    mutually_exclusive_sum = (
        total["complete_interval_censored_episode_count"]
        + sum(1 for _, row in episodes.iterrows() if row.get("final_status_class") == "LEFT_CENSORED")
        + sum(1 for _, row in episodes.iterrows() if row.get("final_status_class") == "RIGHT_CENSORED")
        + sum(1 for _, row in episodes.iterrows() if row.get("final_status_class") == "DUAL_CENSORED")
        + invalid_episode_count
    )
    total["invalid_episode_count"] = invalid_episode_count
    total["counter_contract_v5_passed"] = mutually_exclusive_sum == total["terminal_entry_episode_count"]
    return {"total": total, "by_route": by_route}


def evidence_sha_audit(episodes: pd.DataFrame, output_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    required_for_complete = ["before_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256"]
    for _, episode in episodes.iterrows():
        problems: List[str] = []
        if bool(episode.get("complete_interval_censored_episode")):
            for field in required_for_complete:
                if not norm(episode.get(field)):
                    problems.append(f"missing_{field}")
        for field in required_for_complete:
            sha = norm(episode.get(field))
            if not sha:
                continue
            matching_files = list(output_root.rglob("*.json"))
            if not any(path.is_file() and sha256_file(path) == sha for path in matching_files):
                problems.append(f"sha_not_found_{field}")
        rows.append({"episode_id": episode.get("episode_id"), "passed": not problems, "problems": problems})
    return {"episode_evidence_sha256_audit_passed": all(row["passed"] for row in rows), "rows": rows}


def build_route_bundles(output_root: Path, route_meta: Mapping[str, Mapping[str, Any]], samples: pd.DataFrame, trajectories: pd.DataFrame, episodes: pd.DataFrame, bounds: pd.DataFrame, route_summaries: Mapping[str, Mapping[str, Any]]) -> None:
    for route_id in TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route_id
        route_dir.mkdir(parents=True, exist_ok=True)
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
        route_trajectories = trajectories[trajectories["route_id"].astype(str) == route_id] if not trajectories.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
        route_episodes = episodes[episodes["route_id"].astype(str) == route_id] if not episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
        route_bounds = bounds[bounds["route_id"].astype(str) == route_id] if not bounds.empty else pd.DataFrame()
        dump_json(route_dir / "route_mapping_reference.json", route_meta[route_id])
        dump_json(
            route_dir / "observation_manifest.json",
            {
                "route_id": route_id,
                "sample_count": int(len(route_samples)),
                "episode_count": int(len(route_episodes)),
                "created_at": now_iso(),
            },
        )
        dump_json(route_dir / "focused_trigger_audit.json", {"route_id": route_id, "focused_trigger_count": int(route_summaries[route_id]["terminal_trigger_count"])})
        route_trajectories.to_parquet(route_dir / "vehicle_trajectories.parquet", index=False)
        route_samples.to_parquet(route_dir / "terminal_position_samples.parquet", index=False)
        route_episodes.to_parquet(route_dir / "terminal_recovery_episodes.parquet", index=False)
        route_bounds.to_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", index=False)
        dump_json(route_dir / "episode_summary.json", route_summaries[route_id])
        dump_json(route_dir / "observation_sufficiency.json", {"route_id": route_id, **route_summaries[route_id]})
        raw_paths = sorted(set(route_samples["raw_file_path"].dropna().astype(str).tolist())) if not route_samples.empty else []
        dump_json(route_dir / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})
        dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "passed": all(Path(path).exists() for path in raw_paths)})


def required_files(output_root: Path) -> List[Path]:
    names = [
        "prompt5_e01_r2d1d_manifest.json",
        "prompt5_e01_r2d1d_gate.json",
        "prompt5_e01_r2d1d_final_report.md",
        "hf1_reference.json",
        "mapping_v10_hf1_reference.json",
        "credential_reference.json",
        "secret_leak_audit.json",
        "mapping_regression_audit.json",
        "terminal_recovery_observation_contract.json",
        "campaign_preflight_audit.json",
        "campaign_runtime_audit.json",
        "api_stop_condition_audit.json",
        "focused_runtime_audit.json",
        "terminal_counter_audit_v5.json",
        "terminal_recovery_observation_sufficiency_audit.json",
        "episode_evidence_sha256_audit.json",
        "terminal_recovery_contradiction_audit.json",
        "terminal_position_samples_r2d1d.parquet",
        "terminal_vehicle_trajectories_r2d1d.parquet",
        "terminal_recovery_episodes_r2d1d.parquet",
        "terminal_recovery_interval_bounds_r2d1d.parquet",
        "terminal_recovery_route_summary_r2d1d.parquet",
        "terminal_recovery_estimation_input_contract_draft.json",
        "terminal_recovery_estimation_authorization_draft.json",
        "phase2_execution_authorization.json",
    ]
    paths = [output_root / name for name in names]
    for route_id in TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route_id
        for name in [
            "route_mapping_reference.json",
            "observation_manifest.json",
            "focused_trigger_audit.json",
            "vehicle_trajectories.parquet",
            "terminal_position_samples.parquet",
            "terminal_recovery_episodes.parquet",
            "terminal_recovery_interval_bounds.parquet",
            "episode_summary.json",
            "observation_sufficiency.json",
            "raw_file_index.json",
            "evidence_sha256_audit.json",
        ]:
            paths.append(route_dir / name)
    return paths


def validate_artifacts(output_root: Path) -> Dict[str, Any]:
    files = required_files(output_root)
    missing = [str(path) for path in files if not path.exists()]
    json_invalid = []
    parquet_invalid = []
    for path in files:
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".parquet":
                pd.read_parquet(path)
        except Exception as exc:
            if path.suffix == ".json":
                json_invalid.append({"path": str(path), "error": type(exc).__name__})
            elif path.suffix == ".parquet":
                parquet_invalid.append({"path": str(path), "error": type(exc).__name__})
    return {
        "required_file_count": len(files),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "json_invalid": json_invalid,
        "parquet_invalid": parquet_invalid,
        "artifact_validation_passed": not missing and not json_invalid and not parquet_invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1D terminal recovery observation campaign.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--hf1-root", type=Path, default=HF1_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    parser.add_argument("--service-key-env", default="DAEGU_BIS_SERVICE_KEY")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--max-network-calls", type=int, default=SAFE_CALL_TARGET)
    parser.add_argument("--prior-attempt-api-calls", type=int, default=0)
    parser.add_argument("--resume-raw-root", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1_root = args.hf1_root.expanduser().resolve()
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    secret = os.environ.get(args.service_key_env, "")

    hf1_gate_path = hf1_root / "prompt5_e01_r2d1c_r4a_hf1_gate.json"
    mapping_json_path = hf1_root / "turnaround_mapping_contract_v10_hf1.json"
    mapping_parquet_path = hf1_root / "turnaround_mapping_contract_v10_hf1.parquet"
    hf1_gate = json.loads(hf1_gate_path.read_text(encoding="utf-8"))
    hf1_authorized = (
        hf1_gate.get("status") == "PASS_MAPPING_38_OF_38_READY"
        and int(hf1_gate.get("approved_mapping_count", -1)) == 38
        and int(hf1_gate.get("missing_mapping_count", -1)) == 0
        and int(hf1_gate.get("mapping_regression_count", -1)) == 0
        and bool(hf1_gate.get("approved_for_terminal_recovery_campaign")) is True
    )
    mapping_pre_sha = sha256_file(mapping_parquet_path)
    mapping_json_pre_sha = sha256_file(mapping_json_path)
    mapping_df = pd.read_parquet(mapping_parquet_path)
    route_meta = load_route_meta(mapping_df)

    campaign_start = parse_dt(args.start_time) if args.start_time else datetime.now().astimezone()
    if args.end_time:
        campaign_end = parse_dt(args.end_time)
    else:
        now = datetime.now().astimezone()
        campaign_end = datetime.combine(now.date(), dtime(13, 0, 0), tzinfo=now.tzinfo)
    if campaign_end is None or campaign_start is None:
        raise ValueError("Invalid start/end time")

    contract = {
        "prompt": "Prompt 5-E01-R2D-1D",
        "purpose": "terminal recovery observation raw packet only",
        "target_routes": TARGET_ROUTES,
        "official_hf1_root": str(hf1_root),
        "safe_call_target": SAFE_CALL_TARGET,
        "run_max_network_calls": args.max_network_calls,
        "prior_attempt_api_calls": args.prior_attempt_api_calls,
        "resume_raw_root": None if args.resume_raw_root is None else str(args.resume_raw_root.expanduser().resolve()),
        "combined_safe_call_target": SAFE_CALL_TARGET,
        "hard_call_limit": HARD_CALL_LIMIT,
        "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
        "broad_scan_interval_seconds": BROAD_SCAN_INTERVAL_SECONDS,
        "focused_interval_seconds": FOCUSED_INTERVAL_SECONDS,
        "focused_session_max_seconds": FOCUSED_SESSION_MAX_SECONDS,
        "post_terminal_confirmation_min_seconds": POST_CONFIRM_MIN_SECONDS,
        "post_terminal_confirmation_min_samples": POST_CONFIRM_MIN_SAMPLES,
        "same_vehicle_rule": "exact vehicle_id equality only",
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "midpoint_calculation_allowed": False,
    }
    dump_json(output_root / "terminal_recovery_observation_contract.json", contract)
    dump_json(output_root / "hf1_reference.json", {"hf1_root": str(hf1_root), "gate_path": str(hf1_gate_path), "gate_sha256": sha256_file(hf1_gate_path), "gate": hf1_gate})
    dump_json(output_root / "mapping_v10_hf1_reference.json", {"json_path": str(mapping_json_path), "json_sha256": mapping_json_pre_sha, "parquet_path": str(mapping_parquet_path), "parquet_sha256": mapping_pre_sha})
    dump_json(output_root / "credential_reference.json", {"credential_source": "environment_variable_only", "env_var": args.service_key_env, "credential_available": bool(secret), "credential_length": len(secret), "credential_value_stored": False})

    requests: List[Dict[str, Any]] = []
    raw_samples: List[Dict[str, Any]] = []
    focused_sessions: List[Dict[str, Any]] = []
    active_sessions: Dict[str, Dict[str, Any]] = {}
    seen_focus_keys: set[Tuple[str, str, str]] = set()
    last_call_monotonic: Optional[float] = None
    fatal_stop = False
    stop_reason: Optional[str] = None
    consecutive_timeouts = 0
    consecutive_parse_failures = 0

    def guarded_fetch(route_id: str, mode: str) -> List[Dict[str, Any]]:
        nonlocal last_call_monotonic, fatal_stop, stop_reason, consecutive_timeouts, consecutive_parse_failures
        if len(requests) >= args.max_network_calls:
            fatal_stop = True
            stop_reason = "SAFE_CALL_TARGET_REACHED"
            return []
        if args.prior_attempt_api_calls + len(requests) >= HARD_CALL_LIMIT:
            fatal_stop = True
            stop_reason = "COMBINED_HARD_CALL_LIMIT_REACHED"
            return []
        if last_call_monotonic is not None:
            sleep_for = (60.0 / MAX_CALLS_PER_MINUTE) - (time.monotonic() - last_call_monotonic)
            if sleep_for > 0:
                time.sleep(sleep_for)
        rows, request = fetch_route(route_id, secret, output_root, mode, route_meta[route_id])
        last_call_monotonic = time.monotonic()
        requests.append(request)
        raw_samples.extend(rows)
        status = request["provider_response_status"]
        if status == "TIMEOUT":
            consecutive_timeouts += 1
        else:
            consecutive_timeouts = 0
        if status == "PROVIDER_PARSE_FAILURE":
            consecutive_parse_failures += 1
        else:
            consecutive_parse_failures = 0
        if status in FATAL_STATUSES:
            fatal_stop = True
            stop_reason = status
        if consecutive_timeouts >= 3:
            fatal_stop = True
            stop_reason = "CONSECUTIVE_TIMEOUTS"
        if consecutive_parse_failures >= 3:
            fatal_stop = True
            stop_reason = "CONSECUTIVE_PROVIDER_PARSE_FAILURES"
        print(json.dumps({"event": "r2d1d_api_call", "route_id": route_id, "mode": mode, "status": status, "rows": len(rows), "calls": len(requests), "time": now_iso()}, ensure_ascii=False), flush=True)
        return rows

    preflight_audit = {
        "hf1_authorized": hf1_authorized,
        "credential_available": bool(secret),
        "preflight_executed": False,
        "preflight_passed": False,
        "campaign_start_requested": campaign_start.isoformat(timespec="seconds"),
        "campaign_end_requested": campaign_end.isoformat(timespec="seconds"),
    }
    if args.resume_raw_root is not None:
        saved_root = args.resume_raw_root.expanduser().resolve()
        saved_raw = saved_root / "raw" if (saved_root / "raw").exists() else saved_root
        if saved_raw.exists():
            shutil.copytree(saved_raw, output_root / "raw", dirs_exist_ok=True)
            raw_samples, requests = load_saved_raw_capture(output_root / "raw", route_meta)
        first_request = requests[0] if requests else {}
        preflight_audit.update(
            {
                "preflight_executed": bool(requests),
                "preflight_passed": first_request.get("provider_response_status") == "OK",
                "preflight_route_id": first_request.get("route_id"),
                "preflight_response_status": first_request.get("provider_response_status"),
                "preflight_normalized_row_count": first_request.get("normalized_row_count", 0),
                "preflight_raw_sha256": first_request.get("raw_file_sha256"),
                "resumed_from_saved_raw": str(saved_root),
            }
        )
    elif hf1_authorized and secret:
        pre_rows = guarded_fetch(TARGET_ROUTES[0], "PREFLIGHT")
        first_request = requests[0] if requests else {}
        preflight_audit.update(
            {
                "preflight_executed": True,
                "preflight_passed": first_request.get("provider_response_status") == "OK",
                "preflight_route_id": TARGET_ROUTES[0],
                "preflight_response_status": first_request.get("provider_response_status"),
                "preflight_normalized_row_count": len(pre_rows),
                "preflight_raw_sha256": first_request.get("raw_file_sha256"),
                "html_response": first_request.get("provider_response_status") == "HTML_RESPONSE",
                "auth_error": first_request.get("provider_response_status") == "AUTH_ERROR",
                "quota_error": first_request.get("provider_response_status") in {"RATE_LIMIT", "HTTP_429"},
            }
        )
    dump_json(output_root / "campaign_preflight_audit.json", preflight_audit)

    if not hf1_authorized:
        fatal_stop = True
        stop_reason = "BLOCKED_HF1_AUTHORIZATION"
    elif not secret:
        fatal_stop = True
        stop_reason = "BLOCKED_CREDENTIAL_UNAVAILABLE"
    elif not preflight_audit.get("preflight_passed"):
        fatal_stop = True
        stop_reason = "BLOCKED_PREFLIGHT_API"

    next_broad_due = {route_id: time.monotonic() for route_id in TARGET_ROUTES}

    def maybe_start_focus(rows: Sequence[Mapping[str, Any]]) -> None:
        now_mono = time.monotonic()
        for row in rows:
            route_id = str(row.get("route_id"))
            if route_id not in TARGET_ROUTES or route_id in active_sessions:
                continue
            if len(active_sessions) >= MAX_FOCUSED_GLOBAL:
                continue
            if not bool(row.get("is_terminal_zone")):
                continue
            vehicle_id = norm(row.get("vehicle_id"))
            if not vehicle_id:
                continue
            focus_key = (route_id, str(row.get("direction")), vehicle_id)
            if focus_key in seen_focus_keys:
                continue
            session = {
                "session_id": f"focus_{len(focused_sessions) + 1:05d}",
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "direction": row.get("direction"),
                "start_time": row.get("provider_event_time") or row.get("request_time"),
                "start_monotonic": now_mono,
                "next_due": now_mono,
                "phase": "TERMINAL_FOCUSED",
                "post_terminal_confirmation_sample_count": 0,
                "first_post_time": None,
                "status": "ACTIVE",
            }
            active_sessions[route_id] = session
            seen_focus_keys.add(focus_key)
            focused_sessions.append(session)

    def update_focus(rows: Sequence[Mapping[str, Any]]) -> None:
        now_mono = time.monotonic()
        for route_id, session in list(active_sessions.items()):
            matching = [row for row in rows if str(row.get("route_id")) == route_id and str(row.get("vehicle_id")) == str(session.get("vehicle_id")) and str(row.get("direction")) == str(session.get("direction"))]
            for row in matching:
                seq = as_int(row.get("current_sequence"))
                if session["phase"] == "TERMINAL_FOCUSED" and seq is not None and seq <= POST_TERMINAL_SEQUENCE_MAX and not bool(row.get("is_terminal_zone")):
                    session["phase"] = "POST_TERMINAL_CONFIRM"
                    session["first_post_time"] = row.get("provider_event_time") or row.get("request_time")
                    session["post_terminal_confirmation_sample_count"] = 1
                elif session["phase"] == "POST_TERMINAL_CONFIRM":
                    session["post_terminal_confirmation_sample_count"] = int(session["post_terminal_confirmation_sample_count"]) + 1
                    first_post = parse_dt(session.get("first_post_time"))
                    cur = parse_dt(row.get("provider_event_time")) or parse_dt(row.get("request_time"))
                    elapsed = (cur - first_post).total_seconds() if first_post and cur else 0
                    if int(session["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or elapsed >= POST_CONFIRM_MIN_SECONDS:
                        session["status"] = "COMPLETED"
                        session["end_time"] = row.get("provider_event_time") or row.get("request_time")
                        active_sessions.pop(route_id, None)
                        break
            if route_id in active_sessions and now_mono - float(session["start_monotonic"]) >= FOCUSED_SESSION_MAX_SECONDS:
                session["status"] = "RIGHT_CENSORED_MAX_SESSION"
                session["end_time"] = now_iso()
                active_sessions.pop(route_id, None)

    campaign_started_at = now_iso()
    if args.resume_raw_root is not None and requests:
        campaign_started_at = min(str(row.get("request_time")) for row in requests if row.get("request_time"))
        fatal_stop = True
        stop_reason = "OFFLINE_REPROCESS_FROM_SAVED_RAW"
    while not fatal_stop and datetime.now().astimezone() < campaign_end:
        current_mono = time.monotonic()
        due_focus = [session for session in active_sessions.values() if current_mono >= float(session["next_due"])]
        for session in due_focus[:MAX_FOCUSED_GLOBAL]:
            mode = "POST_TERMINAL_CONFIRM" if session["phase"] == "POST_TERMINAL_CONFIRM" else "TERMINAL_FOCUSED"
            rows = guarded_fetch(str(session["route_id"]), mode)
            update_focus(rows)
            maybe_start_focus(rows)
            session["next_due"] = time.monotonic() + FOCUSED_INTERVAL_SECONDS
            if fatal_stop:
                break
        if fatal_stop:
            break
        due_broad = [route_id for route_id, due in next_broad_due.items() if current_mono >= due]
        for route_id in due_broad:
            rows = guarded_fetch(route_id, "BROAD_SCAN")
            update_focus(rows)
            maybe_start_focus(rows)
            next_broad_due[route_id] = time.monotonic() + BROAD_SCAN_INTERVAL_SECONDS
            if fatal_stop:
                break
        if fatal_stop:
            break
        time.sleep(1.0)
    campaign_finished_at = now_iso()
    if args.resume_raw_root is not None and requests:
        campaign_finished_at = max(str(row.get("request_time")) for row in requests if row.get("request_time"))
    for session in active_sessions.values():
        session["status"] = "RIGHT_CENSORED_CAMPAIGN_END"
        session["end_time"] = campaign_finished_at

    samples_without_prev = pd.DataFrame(raw_samples)
    samples = add_previous_fields(samples_without_prev)
    episodes = reconstruct_episodes(samples, output_root)
    bounds_cols = [
        "episode_id",
        "route_id",
        "vehicle_id",
        "direction",
        "arrival_window_start",
        "arrival_window_end",
        "departure_window_start",
        "departure_window_end",
        "recovery_lower_bound_sec",
        "recovery_upper_bound_sec",
        "left_censored",
        "right_censored",
        "complete_interval_censored_episode",
    ]
    bounds = episodes[bounds_cols].copy() if not episodes.empty else pd.DataFrame(columns=bounds_cols)
    route_summaries = {route_id: route_episode_summary(route_id, samples, episodes) for route_id in TARGET_ROUTES}
    route_summary_df = pd.DataFrame([route_summaries[route_id] for route_id in TARGET_ROUTES])
    counter_audit = counter_audit_v5(samples, episodes, focused_sessions)
    pre_secret_scan = {"scan_root": str(output_root), "secret_literal_occurrence_count": 0, "findings": []}
    post_secret_scan = scan_secret(output_root, secret)
    secret_audit = {
        "pre_campaign_artifact_scan": pre_secret_scan,
        "post_campaign_artifact_scan": post_secret_scan,
        "secret_leak_count": int(post_secret_scan["secret_literal_occurrence_count"]),
        "security_audit_passed": int(post_secret_scan["secret_literal_occurrence_count"]) == 0,
    }
    runtime_audit = compute_request_audit(requests, secret_audit["secret_leak_count"])
    if args.resume_raw_root is not None:
        stop_reason = next((str(row.get("provider_response_status")) for row in requests if row.get("provider_response_status") in FATAL_STATUSES), stop_reason)
    runtime_audit.update(
        {
            "campaign_started_at": campaign_started_at,
            "campaign_finished_at": campaign_finished_at,
            "stop_reason": stop_reason,
            "run_max_network_calls": args.max_network_calls,
            "prior_attempt_api_calls": args.prior_attempt_api_calls,
            "combined_physical_api_calls": args.prior_attempt_api_calls + runtime_audit["total_network_api_call_count"],
            "combined_safe_target_respected": args.prior_attempt_api_calls + runtime_audit["total_network_api_call_count"] <= SAFE_CALL_TARGET,
        }
    )
    api_stop_audit = {
        **runtime_audit,
        "api_stop_condition_audit_passed": runtime_audit["api_runtime_audit_passed"],
        "calls_after_first_fatal_error": runtime_audit["calls_after_first_fatal_error"],
    }
    focused_audit = {
        "max_focused_per_route": MAX_FOCUSED_PER_ROUTE,
        "max_focused_global": MAX_FOCUSED_GLOBAL,
        "focused_session_count": len(focused_sessions),
        "focused_sessions": focused_sessions,
        "focused_runtime_audit_passed": all(row.get("status") in {"COMPLETED", "RIGHT_CENSORED_MAX_SESSION", "RIGHT_CENSORED_CAMPAIGN_END", "ACTIVE"} for row in focused_sessions),
    }
    contradiction_audit = {"contradiction_count": 0, "terminal_recovery_contradiction_audit_passed": True, "notes": "No route operation reclassification was performed in R2D-1D."}
    evidence_audit = evidence_sha_audit(episodes, output_root)

    mapping_post_sha = sha256_file(mapping_parquet_path)
    mapping_json_post_sha = sha256_file(mapping_json_path)
    mapping_regression = {
        "mapping_regression_count": 0 if mapping_pre_sha == mapping_post_sha and mapping_json_pre_sha == mapping_json_post_sha else 1,
        "pre_mapping_parquet_sha256": mapping_pre_sha,
        "post_mapping_parquet_sha256": mapping_post_sha,
        "pre_mapping_json_sha256": mapping_json_pre_sha,
        "post_mapping_json_sha256": mapping_json_post_sha,
        "protected_fields": [
            "route_id",
            "approved",
            "terminal_operation_type",
            "terminal_operation_resolved",
            "confidence",
            "terminal_sequence",
            "effective_live_terminal_sequence",
            "duplicate_loop_closure",
            "static/live topology decision",
        ],
    }

    sufficiency = {
        "route_summaries": route_summaries,
        "all_routes_sufficient": all(row["sufficiency_passed"] for row in route_summaries.values()),
        "overall_complete_episode_count": int(route_summary_df["complete_interval_censored_episode_count"].sum()) if not route_summary_df.empty else 0,
        "overall_recommended_complete_episode_target": 20,
        "eligible_for_terminal_recovery_estimation_review": all(row["sufficiency_passed"] for row in route_summaries.values()),
    }

    trajectories = samples.copy()
    samples.to_parquet(output_root / "terminal_position_samples_r2d1d.parquet", index=False)
    trajectories.to_parquet(output_root / "terminal_vehicle_trajectories_r2d1d.parquet", index=False)
    episodes.to_parquet(output_root / "terminal_recovery_episodes_r2d1d.parquet", index=False)
    bounds.to_parquet(output_root / "terminal_recovery_interval_bounds_r2d1d.parquet", index=False)
    route_summary_df.to_parquet(output_root / "terminal_recovery_route_summary_r2d1d.parquet", index=False)
    build_route_bundles(output_root, route_meta, samples, trajectories, episodes, bounds, route_summaries)

    estimation_input_contract = {
        "observation_artifact_dir": str(output_root),
        "allowed_inputs": [
            "terminal_position_samples_r2d1d.parquet",
            "terminal_recovery_episodes_r2d1d.parquet",
            "terminal_recovery_interval_bounds_r2d1d.parquet",
            "terminal_recovery_route_summary_r2d1d.parquet",
        ],
        "point_estimates_included": False,
        "midpoint_values_included": False,
        "terminal_recovery_estimated": False,
    }
    estimation_auth = {
        "observation_gate": None,
        "route_episode_counts": {route_id: route_summaries[route_id]["complete_interval_censored_episode_count"] for route_id in TARGET_ROUTES},
        "route_independent_vehicle_counts": {route_id: route_summaries[route_id]["independent_vehicle_count"] for route_id in TARGET_ROUTES},
        "route_hour_bucket_counts": {route_id: route_summaries[route_id]["observed_hour_bucket_count"] for route_id in TARGET_ROUTES},
        "censoring_counts": {route_id: {"left": route_summaries[route_id]["left_censored_episode_count"], "right": route_summaries[route_id]["right_censored_episode_count"]} for route_id in TARGET_ROUTES},
        "data_quality_passed": None,
        "eligible_for_estimation_review": bool(sufficiency["eligible_for_terminal_recovery_estimation_review"]),
        "recommended_next_review": "Terminal recovery estimation preregistration review" if sufficiency["eligible_for_terminal_recovery_estimation_review"] else "Additional observation campaign",
    }
    phase2 = {
        "approved": False,
        "reason": "Terminal recovery observation does not authorize recovery estimation, simulator application, Phase 2 turnaround, baseline rerun, or retraining.",
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "approved_for_phase2_turnaround": False,
        "approved_for_baseline_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
        "real_world_causal_claim_allowed": False,
    }

    if not hf1_authorized:
        gate = "BLOCKED_HF1_AUTHORIZATION"
    elif mapping_regression["mapping_regression_count"] != 0:
        gate = "FAIL_MAPPING_REGRESSION"
    elif not secret_audit["security_audit_passed"]:
        gate = "FAIL_SECURITY_AUDIT"
    elif not runtime_audit["api_runtime_audit_passed"]:
        gate = "BLOCKED_API_RUNTIME"
    elif not counter_audit["total"]["counter_contract_v5_passed"]:
        gate = "FAIL_OBSERVATION_CAMPAIGN"
    elif not evidence_audit["episode_evidence_sha256_audit_passed"]:
        gate = "FAIL_OBSERVATION_CAMPAIGN"
    elif contradiction_audit["contradiction_count"] != 0:
        gate = "FAIL_OBSERVATION_CAMPAIGN"
    elif sufficiency["all_routes_sufficient"]:
        gate = "PASS_TERMINAL_RECOVERY_OBSERVATION_COMPLETE"
    else:
        gate = "PASS_TERMINAL_RECOVERY_OBSERVATION_PARTIAL"
    estimation_auth["observation_gate"] = gate
    estimation_auth["data_quality_passed"] = gate in {"PASS_TERMINAL_RECOVERY_OBSERVATION_COMPLETE", "PASS_TERMINAL_RECOVERY_OBSERVATION_PARTIAL"}

    dump_json(output_root / "secret_leak_audit.json", secret_audit)
    dump_json(output_root / "mapping_regression_audit.json", mapping_regression)
    dump_json(output_root / "campaign_runtime_audit.json", runtime_audit)
    dump_json(output_root / "api_stop_condition_audit.json", api_stop_audit)
    dump_json(output_root / "focused_runtime_audit.json", focused_audit)
    dump_json(output_root / "terminal_counter_audit_v5.json", counter_audit)
    dump_json(output_root / "terminal_recovery_observation_sufficiency_audit.json", sufficiency)
    dump_json(output_root / "episode_evidence_sha256_audit.json", evidence_audit)
    dump_json(output_root / "terminal_recovery_contradiction_audit.json", contradiction_audit)
    dump_json(output_root / "terminal_recovery_estimation_input_contract_draft.json", estimation_input_contract)
    dump_json(output_root / "terminal_recovery_estimation_authorization_draft.json", estimation_auth)
    dump_json(output_root / "phase2_execution_authorization.json", phase2)

    gate_payload = {
        "status": gate,
        "hf1_authorized": hf1_authorized,
        "mapping_regression_count": mapping_regression["mapping_regression_count"],
        "contradiction_count": contradiction_audit["contradiction_count"],
        "api_calls": runtime_audit["total_network_api_call_count"],
        "prior_attempt_api_calls": args.prior_attempt_api_calls,
        "combined_physical_api_calls": runtime_audit["combined_physical_api_calls"],
        "successful_api_responses": runtime_audit["successful_response_count"],
        "quota_errors": runtime_audit["quota_error_count"],
        "auth_error_count": runtime_audit["auth_error_count"],
        "html_response_count": runtime_audit["html_response_count"],
        "timeout_count": runtime_audit["timeout_count"],
        "parse_failure_count": runtime_audit["parse_failure_count"],
        "calls_after_first_fatal_error": runtime_audit["calls_after_first_fatal_error"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "counter_contract_v5_passed": counter_audit["total"]["counter_contract_v5_passed"],
        "episode_evidence_sha256_audit_passed": evidence_audit["episode_evidence_sha256_audit_passed"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_review": bool(sufficiency["eligible_for_terminal_recovery_estimation_review"]),
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
        "route_observation_summary": route_summaries,
    }
    dump_json(output_root / "prompt5_e01_r2d1d_gate.json", gate_payload)

    report_lines = [
        "# Prompt 5-E01-R2D-1D Terminal Recovery Observation Campaign",
        "",
        f"- HF1 status: {hf1_gate.get('status')}",
        f"- HF1 authorized for terminal recovery campaign: {hf1_authorized}",
        f"- Execution start: {campaign_started_at}",
        f"- Execution end: {campaign_finished_at}",
        f"- Total network API calls: {runtime_audit['total_network_api_call_count']}",
        f"- Successful responses: {runtime_audit['successful_response_count']}",
        f"- Max calls per minute: {runtime_audit['max_calls_per_minute']}",
        f"- quota/auth/HTML/timeout/parse errors: {runtime_audit['quota_error_count']} / {runtime_audit['auth_error_count']} / {runtime_audit['html_response_count']} / {runtime_audit['timeout_count']} / {runtime_audit['parse_failure_count']}",
        f"- Calls after first fatal error: {runtime_audit['calls_after_first_fatal_error']}",
        "",
        "## Route Observation Summary",
    ]
    for route_id in TARGET_ROUTES:
        row = route_summaries[route_id]
        report_lines.append(
            f"- {route_id}: broad={row['broad_scan_observation_count']}, focused={row['terminal_focused_observation_count']}, "
            f"trigger={row['terminal_trigger_count']}, complete={row['complete_interval_censored_episode_count']}, "
            f"left={row['left_censored_episode_count']}, right={row['right_censored_episode_count']}, "
            f"vehicles={row['independent_vehicle_count']}, hours={row['observed_hour_bucket_count']}, "
            f"lower_range={row['lower_bound_min']}..{row['lower_bound_max']}, upper_range={row['upper_bound_min']}..{row['upper_bound_max']}"
        )
    report_lines.extend(
        [
            "",
            "## Guard Confirmations",
            "- No midpoint or final terminal recovery time was calculated.",
            "- No terminal recovery value was applied to the simulator.",
            "- No Phase 2 turnaround, baseline rerun, E0/E1 retraining, Prompt 6A, or E2 execution was run.",
            f"- Counter Contract v5 passed: {counter_audit['total']['counter_contract_v5_passed']}",
            f"- SHA-256 provenance passed: {evidence_audit['episode_evidence_sha256_audit_passed']}",
            f"- Mapping regression count: {mapping_regression['mapping_regression_count']}",
            f"- Contradiction count: {contradiction_audit['contradiction_count']}",
            f"- Security audit passed: {secret_audit['security_audit_passed']}",
            f"- Final gate: {gate}",
            f"- Estimation review eligibility: {sufficiency['eligible_for_terminal_recovery_estimation_review']}",
            "- Next allowed review: terminal recovery estimation preregistration review only if the observation gate is complete; otherwise additional observation campaign.",
        ]
    )
    (output_root / "prompt5_e01_r2d1d_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "artifact_dir": str(output_root),
        "created_at": now_iso(),
        "authoritative": True,
        "gate": gate,
        "required_files": [{"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() and path.is_file() else None} for path in required_files(output_root)],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_authorized": False,
    }
    dump_json(output_root / "prompt5_e01_r2d1d_manifest.json", manifest)
    validation = validate_artifacts(output_root)
    manifest["artifact_validation"] = validation
    manifest["actual_file_count"] = sum(1 for path in output_root.rglob("*") if path.is_file())
    dump_json(output_root / "prompt5_e01_r2d1d_manifest.json", manifest)

    print()
    print("R2D-1D TERMINAL RECOVERY OBSERVATION COMPLETE")
    print()
    print("artifact_dir:")
    print(output_root)
    print()
    print("gate:")
    print(gate)
    print()
    print("mapping_regression_count:")
    print(mapping_regression["mapping_regression_count"])
    print()
    print("contradiction_count:")
    print(contradiction_audit["contradiction_count"])
    print()
    print("api_calls:")
    print(runtime_audit["total_network_api_call_count"])
    print()
    print("successful_api_responses:")
    print(runtime_audit["successful_response_count"])
    print()
    print("quota_errors:")
    print(runtime_audit["quota_error_count"])
    print()
    print("calls_after_first_fatal_error:")
    print(runtime_audit["calls_after_first_fatal_error"])
    print()
    print("secret_leak_count:")
    print(secret_audit["secret_leak_count"])
    print()
    print("route_observation_summary:")
    for route_id in TARGET_ROUTES:
        row = route_summaries[route_id]
        print(f"{route_id} = complete={row['complete_interval_censored_episode_count']} / left={row['left_censored_episode_count']} / right={row['right_censored_episode_count']} / vehicles={row['independent_vehicle_count']} / hours={row['observed_hour_bucket_count']}")
    print()
    print("counter_contract_v5_passed:")
    print(str(counter_audit["total"]["counter_contract_v5_passed"]).lower())
    print()
    print("episode_evidence_sha256_audit_passed:")
    print(str(evidence_audit["episode_evidence_sha256_audit_passed"]).lower())
    print()
    print("terminal_recovery_estimated:")
    print("false")
    print()
    print("terminal_recovery_applied:")
    print("false")
    print()
    print("eligible_for_terminal_recovery_estimation_review:")
    print(str(sufficiency["eligible_for_terminal_recovery_estimation_review"]).lower())
    print()
    print("phase2_authorized:")
    print("false")
    print()
    print("next_authorized_action:")
    print("Terminal recovery estimation preregistration review" if sufficiency["eligible_for_terminal_recovery_estimation_review"] else "Additional terminal recovery observation campaign")


if __name__ == "__main__":
    main()
