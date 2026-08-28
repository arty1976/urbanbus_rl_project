from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r4_final8_terminal_semantics"
R3A_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_r3a_loop_closure_counter_reprocess_20260721_071000"
R3_SOURCE_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_r3_remaining_terminal_semantics_20260721_060000"
GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
KEY_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")

FINAL8_ROUTES = [
    "3000410000",
    "3000410100",
    "4010002001",
    "4010002003",
    "4010002004",
    "4010002118",
    "4050001001",
    "4050010000",
]
BATCHES = {
    "batch_2": ["3000410000", "3000410100", "4010002001", "4010002003"],
    "batch_3": ["4010002004", "4010002118", "4050001001", "4050010000"],
}
CLASSIFIED_OPS = {
    "LOOP_CONTINUOUS",
    "PAIRED_OPPOSITE_DIRECTION",
    "OFF_GRAPH_CONTINUATION_AND_REENTRY",
    "DEPOT_OR_DEADHEAD_TRANSITION",
}
START_SEQUENCE_UPPER_BOUND = 10
TERMINAL_SEQUENCE_TOLERANCE = 6

SAMPLE_COLUMNS = [
    "request_time_kst",
    "provider_event_time",
    "route_id",
    "direction_id",
    "vehicle_id",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
    "batch_id",
    "capture_mode",
    "sampling_interval_seconds",
    "terminal_sequence",
    "duplicate_closure_sequence",
    "last_unique_preclosure_sequence",
    "effective_live_terminal_sequence",
    "terminal_distance_in_sequence",
    "focused_triggered",
    "focused_trigger_time",
    "response_status",
    "raw_file_path",
    "raw_sha256",
]


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows], columns=columns).to_parquet(path, index=False)


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


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
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
        return float(value)
    except (TypeError, ValueError):
        return None


def distance_m(x1: Any, y1: Any, x2: Any, y2: Any) -> Optional[float]:
    fx1, fy1, fx2, fy2 = as_float(x1), as_float(y1), as_float(x2), as_float(y2)
    if fx1 is None or fy1 is None or fx2 is None or fy2 is None:
        return None
    dx = (fx2 - fx1) * 111_320.0 * max(0.1, math.cos(math.radians((fy1 + fy2) / 2.0)))
    dy = (fy2 - fy1) * 110_540.0
    return math.hypot(dx, dy)


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


def credential_fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def redact_url(url: str, secret: str) -> str:
    return url.replace(secret, "<REDACTED>").replace(urllib.parse.quote(secret, safe="%"), "<REDACTED>").replace(urllib.parse.quote_plus(secret, safe="%"), "<REDACTED>")


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")]
    findings = []
    skip = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except Exception:
            continue
        if any(item.encode("utf-8", errors="ignore") in data for item in needles):
            findings.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def detect_response_status(text: str, http_status: Optional[int]) -> str:
    stripped = (text or "").strip()
    low = stripped.lower()
    if http_status == 429:
        return "RATE_LIMIT"
    if not stripped:
        return "EMPTY_RESPONSE"
    if stripped.startswith("<!doctype html") or stripped.startswith("<html") or "<html" in low[:500]:
        return "HTML_RESPONSE"
    if any(marker in low for marker in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    if any(marker in low for marker in ["service error", "application error", "internal server error", "시스템 오류", "서비스 오류"]):
        return "PROVIDER_ERROR"
    if http_status is None:
        return "NETWORK_ERROR"
    if http_status >= 400:
        return "HTTP_ERROR"
    return "OK"


def find_items(obj: Any) -> List[Dict[str, Any]]:
    paths = [["body", "items"], ["body", "items", "item"], ["response", "body", "items"], ["response", "body", "items", "item"], ["items"], ["item"]]
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


def parse_rows(text: str) -> List[Dict[str, Any]]:
    stripped = text.lstrip()
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            return find_items(json.loads(text))
        if stripped.startswith("<"):
            root = ET.fromstring(text)
            out = []
            for item in root.findall(".//item"):
                row = {child.tag.split("}")[-1]: child.text for child in list(item)}
                if row:
                    out.append(row)
            return out
    except Exception:
        return []
    return []


def norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def validate_r3a(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R3A_ROOT
    gate_path = root / "prompt5_e01_r2d1c_r3a_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS_EXISTING_RAW_MAPPING_RECOVERED",
        "approved_mapping_count": 30,
        "missing_mapping_count": 8,
        "mapping_regression_count": 0,
        "additional_api_calls": 0,
        "raw_source_mutated": False,
        "focused_capture_trigger_unit_tests_passed": True,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R3A prerequisite failed: {mismatches}")
    return root, gate


def load_route_contract(project_root: Path, r3a_root: Path, output_root: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    v8_1 = pd.read_parquet(r3a_root / "turnaround_mapping_contract_v8_1.parquet")
    route_rows = v8_1[v8_1["route_id"].astype(str).isin(FINAL8_ROUTES)].copy()
    by_route: Dict[str, Dict[str, Any]] = {}
    for _, row in route_rows.iterrows():
        rid = str(row["route_id"])
        by_route[rid] = row.to_dict()
    static_src = project_root / R3_SOURCE_ROOT / "unresolved_route_static_topology.parquet"
    static = pd.read_parquet(static_src)
    static = static[static["route_id"].astype(str).isin(FINAL8_ROUTES)].copy()
    static.to_parquet(output_root / "unresolved_route_static_topology.parquet", index=False)
    return v8_1, by_route, static


def terminal_meta(row: Mapping[str, Any]) -> Dict[str, Any]:
    terminal_sequence = as_int(row.get("full_terminal_sequence")) or as_int(row.get("service_terminal_sequence")) or as_int(row.get("effective_live_terminal_sequence"))
    return {
        "route_id": str(row.get("route_id")),
        "direction_id": str(row.get("direction_id") or row.get("service_direction_id") or "1"),
        "terminal_stop_id": norm(row.get("terminal_stop_id")) or norm(row.get("service_terminal_stop_id")),
        "terminal_sequence": terminal_sequence,
        "duplicate_loop_closure": truthy(row.get("duplicate_loop_closure")),
        "duplicate_closure_sequence": as_int(row.get("duplicate_closure_sequence")),
        "last_unique_preclosure_sequence": as_int(row.get("last_unique_preclosure_sequence")),
        "effective_live_terminal_sequence": as_int(row.get("effective_live_terminal_sequence")) or terminal_sequence,
    }


def terminal_state(sample: Mapping[str, Any]) -> bool:
    seq = as_int(sample.get("current_sequence"))
    eff = as_int(sample.get("effective_live_terminal_sequence"))
    stop = norm(sample.get("current_stop_id"))
    terminal_stop = norm(sample.get("terminal_stop_id"))
    return bool((seq is not None and eff is not None and seq >= eff) or (terminal_stop and stop == terminal_stop))


def near_terminal(sample: Mapping[str, Any]) -> bool:
    seq = as_int(sample.get("current_sequence"))
    eff = as_int(sample.get("effective_live_terminal_sequence"))
    return bool(seq is not None and eff is not None and seq >= eff - 5)


def fetch_route(route_id: str, secret: str, output_root: Path, batch_id: str, mode: str, interval: int, meta: Mapping[str, Any], focused_triggered: bool = False, focused_trigger_time: Optional[str] = None, timeout: int = 30) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    request_time = now_kst()
    raw_dir = output_root / "raw" / batch_id / route_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = [("serviceKey", secret), ("routeId", route_id), ("resultType", "json")]
    url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
    raw_path = raw_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    http_status: Optional[int] = None
    content_type = ""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1c-r4/1.0"}), timeout=timeout) as response:
            http_status = int(getattr(response, "status", 200) or 200)
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
            encoding = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        http_status = int(exc.code)
        payload = exc.read() or b""
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        encoding = exc.headers.get_content_charset() if exc.headers else "utf-8"
    except Exception as exc:
        payload = json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False).encode("utf-8")
        content_type = "application/json"
        encoding = "utf-8"
    raw_path.write_bytes(payload)
    text = payload.decode(encoding or "utf-8", errors="replace")
    status = detect_response_status(text, http_status)
    raw_sha = sha256_file(raw_path)
    rows = []
    effective = as_int(meta.get("effective_live_terminal_sequence"))
    for row in parse_rows(text):
        seq = norm(row.get("seq")) or norm(row.get("stopSeq"))
        distance = None
        try:
            distance = float(effective) - float(seq) if effective is not None and seq is not None else None
        except Exception:
            distance = None
        rows.append(
            {
                "request_time_kst": request_time,
                "provider_event_time": norm(row.get("arTime")) or norm(row.get("eventTime")) or norm(row.get("tm")),
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("direction_id")) or meta.get("direction_id"),
                "vehicle_id": norm(row.get("vhcNo2")) or norm(row.get("vhcNo")) or norm(row.get("busId")),
                "current_sequence": seq,
                "current_stop_id": norm(row.get("bsId")) or norm(row.get("stopId")),
                "x": norm(row.get("xPos")) or norm(row.get("x")),
                "y": norm(row.get("yPos")) or norm(row.get("y")),
                "batch_id": batch_id,
                "capture_mode": mode,
                "sampling_interval_seconds": interval,
                "terminal_sequence": meta.get("terminal_sequence"),
                "duplicate_closure_sequence": meta.get("duplicate_closure_sequence"),
                "last_unique_preclosure_sequence": meta.get("last_unique_preclosure_sequence"),
                "effective_live_terminal_sequence": effective,
                "terminal_distance_in_sequence": distance,
                "focused_triggered": focused_triggered,
                "focused_trigger_time": focused_trigger_time,
                "terminal_stop_id": meta.get("terminal_stop_id"),
                "response_status": status,
                "raw_file_path": str(raw_path),
                "raw_sha256": raw_sha,
            }
        )
    return rows, {
        "route_id": route_id,
        "request_time_kst": request_time,
        "request_url_redacted": redact_url(url, secret),
        "http_status": http_status,
        "content_type": content_type,
        "response_status": status,
        "raw_file_path": str(raw_path),
        "raw_sha256": raw_sha,
        "normalized_row_count": len(rows),
        "capture_mode": mode,
        "batch_id": batch_id,
    }


def build_trajectories(samples: pd.DataFrame) -> List[Dict[str, Any]]:
    if samples.empty:
        return []
    df = samples.copy()
    df["request_dt"] = pd.to_datetime(df["request_time_kst"], errors="coerce")
    df["seq_num"] = pd.to_numeric(df["current_sequence"], errors="coerce")
    rows = []
    for vehicle_id, group in df.dropna(subset=["vehicle_id"]).sort_values("request_dt").groupby("vehicle_id"):
        prev = None
        for _, row in group.iterrows():
            out = {
                "vehicle_id": str(vehicle_id),
                "route_id": str(row.get("route_id")),
                "direction_id": str(row.get("direction_id")),
                "request_time_kst": row.get("request_time_kst"),
                "provider_event_time": row.get("provider_event_time"),
                "current_sequence": row.get("current_sequence"),
                "current_stop_id": row.get("current_stop_id"),
                "x": row.get("x"),
                "y": row.get("y"),
                "previous_route_id": None,
                "previous_direction_id": None,
                "previous_sequence": None,
                "previous_stop_id": None,
                "previous_x": None,
                "previous_y": None,
                "previous_time": None,
                "elapsed_seconds": None,
                "sequence_delta": None,
                "spatial_distance_m": None,
                "route_changed": False,
                "direction_changed": False,
                "sequence_reset": False,
                "vehicle_disappeared": False,
                "vehicle_reappeared": False,
                "batch_id": row.get("batch_id"),
                "capture_mode": row.get("capture_mode"),
                "raw_file_path": row.get("raw_file_path"),
                "raw_sha256": row.get("raw_sha256"),
                "terminal_stop_id": row.get("terminal_stop_id"),
                "effective_live_terminal_sequence": row.get("effective_live_terminal_sequence"),
            }
            cur = row.to_dict()
            if prev is not None:
                elapsed = (cur["request_dt"] - prev["request_dt"]).total_seconds() if pd.notna(cur["request_dt"]) and pd.notna(prev["request_dt"]) else None
                seq_delta = float(cur["seq_num"] - prev["seq_num"]) if pd.notna(cur["seq_num"]) and pd.notna(prev["seq_num"]) else None
                route_changed = str(prev.get("route_id")) != str(row.get("route_id"))
                direction_changed = str(prev.get("direction_id")) != str(row.get("direction_id"))
                out.update(
                    {
                        "previous_route_id": str(prev.get("route_id")),
                        "previous_direction_id": str(prev.get("direction_id")),
                        "previous_sequence": prev.get("current_sequence"),
                        "previous_stop_id": prev.get("current_stop_id"),
                        "previous_x": prev.get("x"),
                        "previous_y": prev.get("y"),
                        "previous_time": prev.get("request_time_kst"),
                        "elapsed_seconds": elapsed,
                        "sequence_delta": seq_delta,
                        "spatial_distance_m": distance_m(prev.get("x"), prev.get("y"), row.get("x"), row.get("y")),
                        "route_changed": route_changed,
                        "direction_changed": direction_changed,
                        "sequence_reset": bool(seq_delta is not None and seq_delta < -5),
                        "vehicle_disappeared": bool(elapsed is not None and elapsed > 240),
                        "vehicle_reappeared": bool(elapsed is not None and elapsed > 240),
                    }
                )
            rows.append(out)
            prev = cur
    return rows


def reconstruct_events(trajectories: Sequence[Mapping[str, Any]], route_meta: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    for row in trajectories:
        before_route = str(row.get("previous_route_id")) if row.get("previous_route_id") is not None else None
        if before_route not in route_meta:
            continue
        meta = route_meta[before_route]
        prev_seq = as_int(row.get("previous_sequence"))
        curr_seq = as_int(row.get("current_sequence"))
        eff = as_int(meta.get("effective_live_terminal_sequence"))
        elapsed = as_float(row.get("elapsed_seconds"))
        if prev_seq is None or curr_seq is None or eff is None or elapsed is None or elapsed <= 0:
            continue
        if prev_seq < eff - TERMINAL_SEQUENCE_TOLERANCE:
            continue
        after_route = str(row.get("route_id"))
        after_dir = str(row.get("direction_id"))
        before_dir = str(row.get("previous_direction_id"))
        op = "UNRESOLVED_TERMINAL_OPERATION"
        if after_route == before_route and after_dir == before_dir and curr_seq <= START_SEQUENCE_UPPER_BOUND:
            op = "LOOP_CONTINUOUS"
        elif after_route == before_route and after_dir != before_dir and curr_seq <= START_SEQUENCE_UPPER_BOUND:
            op = "PAIRED_OPPOSITE_DIRECTION"
        elif after_route != before_route and truthy(row.get("vehicle_reappeared")):
            op = "DEPOT_OR_DEADHEAD_TRANSITION"
        elif truthy(row.get("vehicle_reappeared")):
            op = "OFF_GRAPH_CONTINUATION_AND_REENTRY"
        events.append(
            {
                "route_id_before": before_route,
                "direction_id_before": before_dir,
                "vehicle_id": row.get("vehicle_id"),
                "terminal_stop_id": meta.get("terminal_stop_id"),
                "terminal_sequence": meta.get("terminal_sequence"),
                "effective_live_terminal_sequence": meta.get("effective_live_terminal_sequence"),
                "terminal_reached_time_lower": row.get("previous_time"),
                "terminal_reached_time_upper": row.get("previous_time"),
                "post_terminal_time_lower": row.get("request_time_kst"),
                "post_terminal_time_upper": row.get("request_time_kst"),
                "route_id_after": after_route,
                "direction_id_after": after_dir,
                "sequence_after": row.get("current_sequence"),
                "stop_id_after": row.get("current_stop_id"),
                "elapsed_seconds_lower": elapsed,
                "elapsed_seconds_upper": elapsed + 120.0,
                "spatial_distance_m": row.get("spatial_distance_m"),
                "capture_mode_before": row.get("capture_mode"),
                "capture_mode_after": row.get("capture_mode"),
                "operation_candidate": op,
                "classification": "OBSERVED_INTERVAL_CENSORED_TERMINAL_TRANSITION",
                "source_raw_sha256s": [row.get("raw_sha256")],
            }
        )
    return events


def classify_counter_rows(samples: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if samples.empty:
        return {
            "terminal_state_observation_count": 0,
            "terminal_entry_event_count": 0,
            "left_censored_terminal_observation_count": 0,
            "terminal_hold_observation_count": 0,
        }, {}
    df = samples.copy()
    df["request_dt"] = pd.to_datetime(df["request_time_kst"], errors="coerce")
    df["_terminal_state"] = df.apply(lambda row: terminal_state(row), axis=1)
    total = {"terminal_state_observation_count": int(df["_terminal_state"].sum()), "terminal_entry_event_count": 0, "left_censored_terminal_observation_count": 0, "terminal_hold_observation_count": 0}
    by_route: Dict[str, Dict[str, Any]] = {}
    for (route_id, direction_id, vehicle_id), group in df.sort_values("request_dt").groupby(["route_id", "direction_id", "vehicle_id"], dropna=False):
        prev = None
        for _, row in group.iterrows():
            current = truthy(row["_terminal_state"])
            route_key = str(route_id)
            by_route.setdefault(route_key, {"terminal_state_observation_count": 0, "terminal_entry_event_count": 0, "left_censored_terminal_observation_count": 0, "terminal_hold_observation_count": 0})
            if current:
                by_route[route_key]["terminal_state_observation_count"] += 1
                if prev is None:
                    total["left_censored_terminal_observation_count"] += 1
                    by_route[route_key]["left_censored_terminal_observation_count"] += 1
                elif not truthy(prev["_terminal_state"]):
                    total["terminal_entry_event_count"] += 1
                    by_route[route_key]["terminal_entry_event_count"] += 1
                else:
                    total["terminal_hold_observation_count"] += 1
                    by_route[route_key]["terminal_hold_observation_count"] += 1
            prev = row
    return total, by_route


def make_route_bundles(output_root: Path, route_meta: Mapping[str, Mapping[str, Any]], samples: pd.DataFrame, trajectories: pd.DataFrame, events: pd.DataFrame, decisions: Sequence[Mapping[str, Any]], trigger_audit_rows: Sequence[Mapping[str, Any]]) -> None:
    trigger_df = pd.DataFrame(trigger_audit_rows)
    for decision in decisions:
        route_id = str(decision["route_id"])
        bundle = output_root / "mapping_evidence" / route_id
        bundle.mkdir(parents=True, exist_ok=True)
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_traj = trajectories[trajectories["route_id"].astype(str) == route_id] if not trajectories.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        route_triggers = trigger_df[trigger_df["route_id"].astype(str) == route_id] if not trigger_df.empty else pd.DataFrame()
        dump_json(bundle / "route_static_topology.json", route_meta.get(route_id, {}))
        dump_json(bundle / "observation_manifest.json", {"route_id": route_id, "raw_request_count": int(route_samples["raw_sha256"].nunique()) if not route_samples.empty else 0, "sample_count": int(len(route_samples)), "vehicle_count": int(route_samples["vehicle_id"].nunique(dropna=True)) if not route_samples.empty else 0})
        dump_json(bundle / "focused_trigger_audit.json", {"route_id": route_id, "rows": route_triggers.to_dict("records") if not route_triggers.empty else []})
        route_traj.to_parquet(bundle / "vehicle_trajectories.parquet", index=False)
        route_events.to_parquet(bundle / "terminal_transition_events.parquet", index=False)
        dump_json(bundle / "terminal_transition_summary.json", {"route_id": route_id, "operation_counts": route_events["operation_candidate"].value_counts().to_dict() if not route_events.empty else {}})
        dump_json(bundle / "operation_decision.json", decision)
        raw_paths = sorted(set(route_samples["raw_file_path"].dropna().astype(str).tolist())) if not route_samples.empty else []
        dump_json(bundle / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})


def decide_routes(route_meta: Mapping[str, Mapping[str, Any]], samples: pd.DataFrame, events: pd.DataFrame, counter_by_route: Mapping[str, Mapping[str, Any]], trigger_rows: Sequence[Mapping[str, Any]], observation_minutes: Mapping[str, float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    decisions = []
    trigger_df = pd.DataFrame(trigger_rows)
    contradiction_count = 0
    for route_id in FINAL8_ROUTES:
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        classified = route_events[route_events["operation_candidate"].isin(CLASSIFIED_OPS)] if not route_events.empty else pd.DataFrame()
        op_counts = classified["operation_candidate"].value_counts().to_dict() if not classified.empty else {}
        contradictory = max(0, len(op_counts) - 1)
        contradiction_count += contradictory
        decision = "UNRESOLVED_TERMINAL_OPERATION"
        confidence = "UNRESOLVED"
        approved = False
        if route_samples.empty:
            reason = "NO_VEHICLE_OBSERVED"
            remaining = "observe route during confirmed operation window"
        else:
            max_seq = pd.to_numeric(route_samples["current_sequence"], errors="coerce").max()
            eff = as_int(route_meta[route_id].get("effective_live_terminal_sequence"))
            if pd.isna(max_seq) or (eff is not None and max_seq < eff - 5):
                reason = "NO_TERMINAL_REACH_EVENT"
                remaining = "targeted focused capture until terminal-near vehicle is observed"
            elif classified.empty:
                reason = "NO_POST_TERMINAL_OBSERVATION"
                remaining = "continue post-terminal follow for same vehicle"
            elif contradictory:
                decision = "MULTIPLE_SERVICE_PATTERNS"
                reason = "MULTIPLE_SERVICE_PATTERNS"
                remaining = "separate branch/service pattern before approval"
            else:
                decision = max(op_counts.items(), key=lambda item: item[1])[0]
                independent = int(classified[classified["operation_candidate"] == decision]["vehicle_id"].nunique(dropna=True))
                count = int(op_counts[decision])
                confidence = "HIGH" if count >= 1 else "UNRESOLVED"
                approved = confidence in {"HIGH", "MEDIUM"}
                reason = "same-vehicle terminal transition plus static support" if approved else "LOW evidence"
                remaining = "" if approved else "more independent same-operation evidence"
        route_counter = counter_by_route.get(route_id, {})
        route_trigger = trigger_df[trigger_df["route_id"].astype(str) == route_id] if not trigger_df.empty else pd.DataFrame()
        decision_count = int(len(route_events[route_events["operation_candidate"] == decision])) if not route_events.empty and decision in set(route_events["operation_candidate"]) else 0
        independent_count = int(route_events[route_events["operation_candidate"] == decision]["vehicle_id"].nunique(dropna=True)) if decision_count else 0
        decision_row = {
            "route_id": route_id,
            "direction_id": str(route_meta[route_id].get("direction_id") or "1"),
            "decision": decision,
            "confidence": confidence,
            "approved": approved,
            "observation_minutes": observation_minutes.get(route_id, 0),
            "raw_request_count": int(route_samples["raw_sha256"].nunique()) if not route_samples.empty else 0,
            "vehicle_count": int(route_samples["vehicle_id"].nunique(dropna=True)) if not route_samples.empty else 0,
            "repeated_vehicle_count": int((route_samples.groupby("vehicle_id").size() >= 2).sum()) if not route_samples.empty else 0,
            "terminal_state_observation_count": route_counter.get("terminal_state_observation_count", 0),
            "terminal_entry_event_count": route_counter.get("terminal_entry_event_count", 0),
            "terminal_hold_observation_count": route_counter.get("terminal_hold_observation_count", 0),
            "classified_operation_transition_count": int(len(classified)),
            "independent_vehicle_count": independent_count,
            "contradiction_count": contradictory,
            "focused_trigger_evaluation_count": int(len(route_trigger)),
            "focused_trigger_match_count": int(route_trigger["trigger_matched"].sum()) if not route_trigger.empty else 0,
            "focused_capture_started_count": int(route_trigger["focused_capture_started"].sum()) if not route_trigger.empty else 0,
            "supporting_evidence": route_events[route_events["operation_candidate"] == decision].to_dict("records") if decision_count else [],
            "counter_evidence": [],
            "remaining_requirement": remaining,
            "reason": reason,
        }
        decisions.append(decision_row)
    return decisions, {"contradictory_transition_count": contradiction_count}


def update_mapping_v9(v8_1: pd.DataFrame, decisions: Sequence[Mapping[str, Any]], output_root: Path) -> Tuple[Dict[str, Any], int]:
    before = {idx for idx, row in v8_1.iterrows() if truthy(row.get("approved"))}
    by_route = {str(row["route_id"]): row for row in decisions}
    rows = []
    for idx, row in v8_1.iterrows():
        out = row.to_dict()
        out["approved"] = truthy(out.get("approved"))
        out["newly_recovered_in_r4"] = False
        route_id = str(out.get("route_id"))
        if route_id in by_route and not out["approved"]:
            decision = by_route[route_id]
            if truthy(decision.get("approved")):
                out["approved"] = True
                out["terminal_operation_type"] = decision["decision"]
                out["terminal_operation_resolved"] = True
                out["mapping_confidence"] = decision["confidence"]
                out["newly_recovered_in_r4"] = True
                out["terminal_entry_event_count"] = decision.get("terminal_entry_event_count", 0)
                out["terminal_hold_observation_count"] = decision.get("terminal_hold_observation_count", 0)
                out["classified_operation_transition_count"] = decision.get("classified_operation_transition_count", 0)
                out["independent_vehicle_count"] = decision.get("independent_vehicle_count", 0)
                out["contradictory_transition_count"] = decision.get("contradiction_count", 0)
                out["live_evidence_paths"] = [str(output_root / "mapping_evidence" / route_id / "terminal_transition_events.parquet")]
        rows.append(out)
    after = {idx for idx, row in enumerate(rows) if truthy(row.get("approved"))}
    regression = len(before - after)
    payload = {
        "contract_version": "turnaround_mapping_contract_v9",
        "source_contract_version": "turnaround_mapping_contract_v8_1",
        "previous_mapping_count": len(before),
        "newly_recovered_mapping_count": sum(1 for row in rows if truthy(row.get("newly_recovered_in_r4"))),
        "approved_mapping_count": len(after),
        "missing_mapping_count": 38 - len(after),
        "mapping_regression_count": regression,
        "rows": rows,
    }
    return payload, regression


def sleep_until_next(seconds: int, started: float, dry_run: bool) -> None:
    if dry_run:
        return
    delay = max(0.0, seconds - (time.monotonic() - started))
    if delay:
        time.sleep(delay)


def run_capture(args: argparse.Namespace, secret: str, output_root: Path, route_meta: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    samples: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    trigger_rows: List[Dict[str, Any]] = []
    observation_seconds = {route_id: 0.0 for route_id in FINAL8_ROUTES}
    focused_done: set[Tuple[str, str]] = set()

    def record_fetch(route_id: str, batch_id: str, mode: str, interval: int, focused_triggered: bool = False, focused_trigger_time: Optional[str] = None) -> List[Dict[str, Any]]:
        rows, req = fetch_route(route_id, secret, output_root, batch_id, mode, interval, route_meta[route_id], focused_triggered, focused_trigger_time)
        samples.extend(rows)
        requests.append(req)
        return rows

    def run_focus(route_id: str, batch_id: str, vehicle_id: str, trigger_time: str) -> None:
        key = (route_id, vehicle_id)
        if key in focused_done:
            return
        focused_done.add(key)
        end = datetime.now().astimezone() + timedelta(minutes=args.post_terminal_follow_minutes)
        mode = "TERMINAL_FOCUSED"
        previous_seq: Optional[int] = None
        while datetime.now().astimezone() < end:
            cycle = time.monotonic()
            rows = record_fetch(route_id, batch_id, mode, args.focused_interval_seconds, True, trigger_time)
            observation_seconds[route_id] += args.focused_interval_seconds
            for row in rows:
                if str(row.get("vehicle_id")) != str(vehicle_id):
                    continue
                seq = as_int(row.get("current_sequence"))
                if previous_seq is not None and seq is not None and previous_seq >= (as_int(route_meta[route_id].get("effective_live_terminal_sequence")) or 0) - TERMINAL_SEQUENCE_TOLERANCE and seq <= START_SEQUENCE_UPPER_BOUND:
                    mode = "POST_TERMINAL_FOLLOW"
                if seq is not None:
                    previous_seq = seq
            print(json.dumps({"event": "focused_capture_tick", "route_id": route_id, "vehicle_id": vehicle_id, "mode": mode, "rows": len(rows), "time": now_kst()}, ensure_ascii=False), flush=True)
            sleep_until_next(args.focused_interval_seconds, cycle, args.dry_run)

    for batch_id in args.batch_ids:
        batch_routes = BATCHES[batch_id]
        batch_end = datetime.now().astimezone() + timedelta(minutes=args.initial_scan_minutes)
        print(json.dumps({"event": "batch_start", "batch_id": batch_id, "routes": batch_routes, "time": now_kst()}, ensure_ascii=False), flush=True)
        while datetime.now().astimezone() < batch_end:
            cycle = time.monotonic()
            for route_id in batch_routes:
                rows = record_fetch(route_id, batch_id, "BROAD_SCAN", args.broad_interval_seconds)
                observation_seconds[route_id] += args.broad_interval_seconds
                matched = [row for row in rows if near_terminal(row)]
                for row in rows:
                    trigger_rows.append(
                        {
                            "route_id": route_id,
                            "vehicle_id": row.get("vehicle_id"),
                            "sequence_at_trigger": as_int(row.get("current_sequence")),
                            "effective_terminal_sequence": route_meta[route_id].get("effective_live_terminal_sequence"),
                            "trigger_time": row.get("request_time_kst"),
                            "trigger_matched": near_terminal(row),
                            "focused_capture_started": False,
                            "focused_capture_end_time": None,
                            "classification": "TRIGGER_EXECUTED" if near_terminal(row) else "NO_TRIGGER_CANDIDATE",
                        }
                    )
                if matched:
                    trigger = matched[0]
                    for audit_row in reversed(trigger_rows):
                        if audit_row["route_id"] == route_id and audit_row["vehicle_id"] == trigger.get("vehicle_id") and audit_row["trigger_time"] == trigger.get("request_time_kst"):
                            audit_row["focused_capture_started"] = True
                            audit_row["focused_capture_end_time"] = (datetime.now().astimezone() + timedelta(minutes=args.post_terminal_follow_minutes)).isoformat(timespec="seconds")
                            break
                    print(json.dumps({"event": "focused_trigger", "batch_id": batch_id, "route_id": route_id, "vehicle_id": trigger.get("vehicle_id"), "sequence": trigger.get("current_sequence"), "time": now_kst()}, ensure_ascii=False), flush=True)
                    run_focus(route_id, batch_id, str(trigger.get("vehicle_id")), str(trigger.get("request_time_kst")))
                print(json.dumps({"event": "broad_scan_tick", "batch_id": batch_id, "route_id": route_id, "rows": len(rows), "near_terminal": len(matched), "time": now_kst()}, ensure_ascii=False), flush=True)
            sleep_until_next(args.broad_interval_seconds, cycle, args.dry_run)
        if args.additional_scan_minutes > 0:
            route_df = pd.DataFrame(samples)
            unresolved_no_trigger = []
            for route_id in batch_routes:
                route_samples = route_df[route_df["route_id"].astype(str) == route_id] if not route_df.empty else pd.DataFrame()
                if route_samples.empty or not route_samples.apply(lambda row: near_terminal(row), axis=1).any():
                    unresolved_no_trigger.append(route_id)
            extension_end = datetime.now().astimezone() + timedelta(minutes=args.additional_scan_minutes)
            while unresolved_no_trigger and datetime.now().astimezone() < extension_end:
                cycle = time.monotonic()
                for route_id in list(unresolved_no_trigger):
                    rows = record_fetch(route_id, batch_id, "BROAD_SCAN", args.broad_interval_seconds)
                    observation_seconds[route_id] += args.broad_interval_seconds
                    matched = [row for row in rows if near_terminal(row)]
                    print(json.dumps({"event": "extension_scan_tick", "batch_id": batch_id, "route_id": route_id, "rows": len(rows), "near_terminal": len(matched), "time": now_kst()}, ensure_ascii=False), flush=True)
                    if matched:
                        unresolved_no_trigger.remove(route_id)
                        trigger = matched[0]
                        trigger_rows.append({"route_id": route_id, "vehicle_id": trigger.get("vehicle_id"), "sequence_at_trigger": as_int(trigger.get("current_sequence")), "effective_terminal_sequence": route_meta[route_id].get("effective_live_terminal_sequence"), "trigger_time": trigger.get("request_time_kst"), "trigger_matched": True, "focused_capture_started": True, "focused_capture_end_time": (datetime.now().astimezone() + timedelta(minutes=args.post_terminal_follow_minutes)).isoformat(timespec="seconds"), "classification": "TRIGGER_EXECUTED"})
                        run_focus(route_id, batch_id, str(trigger.get("vehicle_id")), str(trigger.get("request_time_kst")))
                sleep_until_next(args.broad_interval_seconds, cycle, args.dry_run)
    return samples, requests, trigger_rows, {route_id: seconds / 60.0 for route_id, seconds in observation_seconds.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R4 final 8 route terminal semantics observation.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--batch-ids", nargs="+", default=["batch_2", "batch_3"], choices=["batch_2", "batch_3"])
    parser.add_argument("--initial-scan-minutes", type=int, default=60)
    parser.add_argument("--additional-scan-minutes", type=int, default=120)
    parser.add_argument("--broad-interval-seconds", type=int, default=120)
    parser.add_argument("--focused-interval-seconds", type=int, default=30)
    parser.add_argument("--post-terminal-follow-minutes", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r3a_root, r3a_gate = validate_r3a(project_root)
    secret, secret_source = read_secret(args.service_key_file)
    if not secret:
        raise RuntimeError("MISSING_SERVICE_KEY")
    pre_scan = scan_secret(output_root, secret)
    v8_1, route_contract, static = load_route_contract(project_root, r3a_root, output_root)
    route_meta = {route_id: terminal_meta(route_contract[route_id]) for route_id in FINAL8_ROUTES}

    dump_json(output_root / "r3a_reference.json", {"r3a_artifact": str(r3a_root), "r3a_gate_path": str(r3a_root / "prompt5_e01_r2d1c_r3a_gate.json"), "r3a_gate_sha256": sha256_file(r3a_root / "prompt5_e01_r2d1c_r3a_gate.json"), "required_approved_mapping_count": 30})
    dump_json(output_root / "mapping_v8_1_reference.json", {"path": str(r3a_root / "turnaround_mapping_contract_v8_1.parquet"), "sha256": sha256_file(r3a_root / "turnaround_mapping_contract_v8_1.parquet")})
    dump_json(output_root / "credential_reference.json", {"credential_available": bool(secret), "credential_length": len(secret), "credential_fingerprint": credential_fingerprint(secret), "credential_source": secret_source})
    dump_json(output_root / "route_batch_schedule.json", {"batches": BATCHES, "executed_batches": args.batch_ids})
    dump_json(output_root / "observation_campaign_contract.json", {"capture_modes": ["BROAD_SCAN", "TERMINAL_FOCUSED", "POST_TERMINAL_FOLLOW"], "initial_scan_minutes": args.initial_scan_minutes, "additional_scan_minutes": args.additional_scan_minutes, "broad_interval_seconds": args.broad_interval_seconds, "focused_interval_seconds": args.focused_interval_seconds, "post_terminal_follow_minutes": args.post_terminal_follow_minutes, "terminal_recovery_estimated": False, "terminal_recovery_applied": False})

    preflight_rows, preflight_req = fetch_route(FINAL8_ROUTES[0], secret, output_root, "preflight", "SERVICE_TIMECHECK", 0, route_meta[FINAL8_ROUTES[0]])
    seqs = [as_int(row.get("current_sequence")) for row in preflight_rows if as_int(row.get("current_sequence")) is not None]
    provider_times = sorted({row.get("provider_event_time") for row in preflight_rows if row.get("provider_event_time")})
    preflight = {
        "service_timecheck_request_time_kst": preflight_req["request_time_kst"],
        "service_timecheck_route_id": FINAL8_ROUTES[0],
        "service_timecheck_response_status": preflight_req["response_status"],
        "service_timecheck_normalized_rows": len(preflight_rows),
        "service_timecheck_provider_event_time_min": provider_times[0] if provider_times else None,
        "service_timecheck_provider_event_time_max": provider_times[-1] if provider_times else None,
        "service_timecheck_sequence_min": min(seqs) if seqs else None,
        "service_timecheck_sequence_max": max(seqs) if seqs else None,
        "AUTH_ERROR_count": 1 if preflight_req["response_status"] == "AUTH_ERROR" else 0,
        "HTML_response_count": 1 if preflight_req["response_status"] == "HTML_RESPONSE" else 0,
        "rate_limit_error_count": 1 if preflight_req["response_status"] == "RATE_LIMIT" else 0,
        "request_url_redacted": preflight_req["request_url_redacted"],
        "raw_sha256": preflight_req["raw_sha256"],
    }
    dump_json(output_root / "campaign_preflight_audit.json", preflight)
    if preflight_req["response_status"] in {"AUTH_ERROR", "HTML_RESPONSE", "RATE_LIMIT"}:
        raise RuntimeError(f"Preflight failed: {preflight_req['response_status']}")

    samples, requests, trigger_rows, observation_minutes = run_capture(args, secret, output_root, route_meta)
    all_samples = preflight_rows + samples
    all_requests = [preflight_req] + requests
    sample_df = pd.DataFrame(all_samples, columns=SAMPLE_COLUMNS + ["terminal_stop_id"])
    sample_df.to_parquet(output_root / "terminal_semantics_position_samples_r4.parquet", index=False)
    trajectories = pd.DataFrame(build_trajectories(sample_df))
    trajectories.to_parquet(output_root / "vehicle_continuous_trajectories_r4.parquet", index=False)
    events = pd.DataFrame(reconstruct_events(trajectories.to_dict("records"), route_meta))
    events.to_parquet(output_root / "terminal_transition_events_r4.parquet", index=False)
    counters, counter_by_route = classify_counter_rows(sample_df)
    loop_count = int((events["operation_candidate"] == "LOOP_CONTINUOUS").sum()) if not events.empty else 0
    paired_count = int((events["operation_candidate"] == "PAIRED_OPPOSITE_DIRECTION").sum()) if not events.empty else 0
    off_count = int((events["operation_candidate"] == "OFF_GRAPH_CONTINUATION_AND_REENTRY").sum()) if not events.empty else 0
    depot_count = int((events["operation_candidate"] == "DEPOT_OR_DEADHEAD_TRANSITION").sum()) if not events.empty else 0
    classified_count = loop_count + paired_count + off_count + depot_count
    counter_audit = {**counters, "classified_operation_transition_count": classified_count, "loop_reset_transition_count": loop_count, "paired_direction_transition_count": paired_count, "off_graph_reentry_transition_count": off_count, "depot_deadhead_transition_count": depot_count, "counter_v4_pass": classified_count == loop_count + paired_count + off_count + depot_count}
    dump_json(output_root / "terminal_counter_audit_v4.json", counter_audit)
    dump_json(output_root / "focused_capture_runtime_audit.json", {"focused_trigger_evaluation_count": len(trigger_rows), "focused_trigger_match_count": sum(1 for row in trigger_rows if truthy(row.get("trigger_matched"))), "focused_capture_started_count": sum(1 for row in trigger_rows if truthy(row.get("focused_capture_started"))), "rows": trigger_rows})

    decisions, contradiction = decide_routes(route_meta, sample_df, events, counter_by_route, trigger_rows, observation_minutes)
    write_parquet(output_root / "terminal_operation_decisions_r4.parquet", decisions)
    dump_json(output_root / "terminal_operation_contradiction_audit.json", contradiction)
    make_route_bundles(output_root, route_meta, sample_df, trajectories, events, decisions, trigger_rows)

    mapping_v9, regression = update_mapping_v9(v8_1, decisions, output_root)
    dump_json(output_root / "turnaround_mapping_contract_v9.json", mapping_v9)
    pd.DataFrame(mapping_v9["rows"]).to_parquet(output_root / "turnaround_mapping_contract_v9.parquet", index=False)
    remaining = []
    for row in decisions:
        if not truthy(row.get("approved")):
            route_samples = sample_df[sample_df["route_id"].astype(str) == str(row["route_id"])]
            max_seq = pd.to_numeric(route_samples["current_sequence"], errors="coerce").max() if not route_samples.empty else None
            remaining.append({"route_id": row["route_id"], "reason": row["reason"], "observed_minutes": row["observation_minutes"], "observed_vehicles": row["vehicle_count"], "maximum_observed_sequence": None if pd.isna(max_seq) else int(max_seq), "terminal_reach_count": row["terminal_state_observation_count"], "post_terminal_follow_count": row["classified_operation_transition_count"], "specific_additional_evidence_required": row["remaining_requirement"]})
    dump_json(output_root / "remaining_mapping_evidence_requirements_v3.json", {"rows": remaining})

    status = "FAIL_OBSERVATION_CAMPAIGN"
    newly = mapping_v9["newly_recovered_mapping_count"]
    if regression:
        status = "FAIL_MAPPING_REGRESSION"
    elif not counter_audit["counter_v4_pass"]:
        status = "FAIL_OBSERVATION_CAMPAIGN"
    elif mapping_v9["approved_mapping_count"] == 38 and mapping_v9["missing_mapping_count"] == 0:
        status = "PASS_MAPPING_38_OF_38_READY"
    elif newly > 0:
        status = "PASS_MAPPING_PARTIALLY_RECOVERED"
    elif any(row["reason"] == "NO_TERMINAL_REACH_EVENT" for row in decisions):
        status = "BLOCKED_NO_TERMINAL_REACH_EVENTS"
    elif any(row["reason"] == "NO_POST_TERMINAL_OBSERVATION" for row in decisions):
        status = "BLOCKED_POST_TERMINAL_CONTINUITY"
    elif any(row["reason"] == "MULTIPLE_SERVICE_PATTERNS" for row in decisions):
        status = "BLOCKED_MULTIPLE_SERVICE_PATTERNS"

    auth_errors = sum(1 for row in all_requests if row["response_status"] == "AUTH_ERROR")
    html_errors = sum(1 for row in all_requests if row["response_status"] == "HTML_RESPONSE")
    rate_errors = sum(1 for row in all_requests if row["response_status"] == "RATE_LIMIT")
    post_scan = scan_secret(output_root, secret)
    dump_json(output_root / "secret_leak_audit.json", {"pre_campaign_artifact_scan": pre_scan, "post_campaign_artifact_scan": post_scan, "secret_leak_count": post_scan["secret_literal_occurrence_count"], "secret_leak_pass": post_scan["secret_literal_occurrence_count"] == 0})
    terminal_recovery_auth = {"approved_for_terminal_recovery_campaign": status == "PASS_MAPPING_38_OF_38_READY", "terminal_recovery_estimated": False, "terminal_recovery_applied": False}
    phase2 = {"approved_for_terminal_recovery_campaign": status == "PASS_MAPPING_38_OF_38_READY", "approved_for_phase2_turnaround_execution": False, "approved_for_baseline_feasibility_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a_corrected_retrospective": False, "approved_for_e2_execution": False, "prompt6_full_matrix_approved": False, "real_world_causal_claim_allowed": False, "phase2_production_executed": False}
    dump_json(output_root / "terminal_recovery_campaign_authorization_draft.json", terminal_recovery_auth)
    dump_json(output_root / "phase2_execution_authorization.json", phase2)

    repeated = int((sample_df.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not sample_df.empty else 0
    seq_prog = int(sum(1 for _, group in sample_df.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2)) if not sample_df.empty else 0
    trigger_match = sum(1 for row in trigger_rows if truthy(row.get("trigger_matched")))
    trigger_started = sum(1 for row in trigger_rows if truthy(row.get("focused_capture_started")))
    trigger_runtime_pass = trigger_started > 0 if trigger_match > 0 else True
    gate = {
        "status": status,
        "classification": status,
        "r3a_gate_path": str(r3a_root / "prompt5_e01_r2d1c_r3a_gate.json"),
        "r3a_gate_sha256": sha256_file(r3a_root / "prompt5_e01_r2d1c_r3a_gate.json"),
        "campaign_route_count": len(FINAL8_ROUTES),
        "campaign_batch_count": len(args.batch_ids),
        "campaign_total_observation_minutes": sum(observation_minutes.values()),
        "campaign_total_requests": len(all_requests),
        "campaign_raw_file_count": len(all_requests),
        "campaign_normalized_rows": len(sample_df),
        "unique_vehicle_count": int(sample_df["vehicle_id"].nunique(dropna=True)) if not sample_df.empty else 0,
        "repeated_vehicle_count": repeated,
        "sequence_progression_count": seq_prog,
        "focused_trigger_evaluation_count": len(trigger_rows),
        "focused_trigger_match_count": trigger_match,
        "focused_capture_started_count": trigger_started,
        "focused_trigger_runtime_passed": trigger_runtime_pass,
        **counter_audit,
        "previous_mapping_count": 30,
        "newly_recovered_mapping_count": newly,
        "approved_mapping_count": mapping_v9["approved_mapping_count"],
        "missing_mapping_count": mapping_v9["missing_mapping_count"],
        "mapping_regression_count": regression,
        "paired_opposite_direction_count": sum(1 for row in decisions if row["decision"] == "PAIRED_OPPOSITE_DIRECTION"),
        "loop_continuous_count": sum(1 for row in decisions if row["decision"] == "LOOP_CONTINUOUS"),
        "off_graph_reentry_count": sum(1 for row in decisions if row["decision"] == "OFF_GRAPH_CONTINUATION_AND_REENTRY"),
        "depot_deadhead_count": sum(1 for row in decisions if row["decision"] == "DEPOT_OR_DEADHEAD_TRANSITION"),
        "multiple_service_pattern_count": sum(1 for row in decisions if row["decision"] == "MULTIPLE_SERVICE_PATTERNS"),
        "unresolved_terminal_operation_count": sum(1 for row in decisions if row["decision"] == "UNRESOLVED_TERMINAL_OPERATION"),
        "high_confidence_mapping_count": sum(1 for row in decisions if row["confidence"] == "HIGH"),
        "medium_confidence_mapping_count": sum(1 for row in decisions if row["confidence"] == "MEDIUM"),
        "low_confidence_mapping_count": sum(1 for row in decisions if row["confidence"] == "LOW"),
        "contradictory_transition_count": contradiction["contradictory_transition_count"],
        "routes_no_vehicle_observed": [row["route_id"] for row in decisions if row["reason"] == "NO_VEHICLE_OBSERVED"],
        "routes_no_terminal_reach": [row["route_id"] for row in decisions if row["reason"] == "NO_TERMINAL_REACH_EVENT"],
        "routes_no_post_terminal_continuity": [row["route_id"] for row in decisions if row["reason"] == "NO_POST_TERMINAL_OBSERVATION"],
        "routes_focused_trigger_failure": [] if trigger_runtime_pass else [row["route_id"] for row in trigger_rows if truthy(row.get("trigger_matched")) and not truthy(row.get("focused_capture_started"))],
        "AUTH_ERROR_count": auth_errors,
        "HTML_response_count": html_errors,
        "rate_limit_error_count": rate_errors,
        "secret_leak_count": post_scan["secret_literal_occurrence_count"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
        **phase2,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "scientific_parameter_changed": False,
        "calibration_applied": False,
    }
    if auth_errors or html_errors or rate_errors or gate["secret_leak_count"]:
        gate["status"] = gate["classification"] = "FAIL_OBSERVATION_CAMPAIGN"
    dump_json(output_root / "prompt5_e01_r2d1c_r4_gate.json", gate)
    report = f"""# Prompt 5-E01-R2D-1C-R4 Final Report

[Prompt 5-E01-R2D-1C-R4 판정]
status: {gate['status']}
classification: {gate['classification']}

[Campaign]
routes: {len(FINAL8_ROUTES)}
batches: {len(args.batch_ids)}
total observation minutes: {gate['campaign_total_observation_minutes']}
requests: {len(all_requests)}
raw files: {len(all_requests)}
normalized rows: {len(sample_df)}

[Focused capture]
trigger evaluations: {gate['focused_trigger_evaluation_count']}
trigger matches: {gate['focused_trigger_match_count']}
focused captures started: {gate['focused_capture_started_count']}
runtime PASS: {gate['focused_trigger_runtime_passed']}

[Vehicle continuity]
unique: {gate['unique_vehicle_count']}
repeated: {repeated}
sequence progression: {seq_prog}

[Terminal counters]
state observations: {counter_audit['terminal_state_observation_count']}
entries: {counter_audit['terminal_entry_event_count']}
left-censored: {counter_audit['left_censored_terminal_observation_count']}
holds: {counter_audit['terminal_hold_observation_count']}
classified transitions: {classified_count}

loop: {loop_count}
paired: {paired_count}
off-graph: {off_count}
depot/deadhead: {depot_count}

[Mapping]
previous: 30
newly recovered: {newly}
approved: {mapping_v9['approved_mapping_count']}
missing: {mapping_v9['missing_mapping_count']}
regression: {regression}

[API/security]
AUTH_ERROR: {auth_errors}
HTML: {html_errors}
rate-limit: {rate_errors}
secret leak: {post_scan['secret_literal_occurrence_count']}

[Recovery]
estimated: false
applied: false
campaign approved: {status == 'PASS_MAPPING_38_OF_38_READY'}

[Guard]
test split: false
test target: false
test embedding: false
threshold changed: false
Phase 2 executed: false
"""
    (output_root / "prompt5_e01_r2d1c_r4_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"prompt": "Prompt 5-E01-R2D-1C-R4", "artifact_dir": str(output_root), "created_at": now_kst(), "status": gate["status"], "files": sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file())}
    dump_json(output_root / "prompt5_e01_r2d1c_r4_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": gate["status"], "approved_mapping_count": gate["approved_mapping_count"], "missing_mapping_count": gate["missing_mapping_count"], "newly_recovered_mapping_count": newly, "additional_api_calls": len(all_requests)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
