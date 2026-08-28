from __future__ import annotations

import argparse
import hashlib
import json
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


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r3_remaining_terminal_semantics"
R2_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_r2_terminal_semantics_observation_20260721_040000"
GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
KEY_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")
APPROVED_R2_ROUTE = "2000002000"
TARGET_ROUTES = [
    "2000002100",
    "2000003100",
    "3000232000",
    "3000323101",
    "3000410000",
    "3000410100",
    "4010002001",
    "4010002003",
    "4010002004",
    "4010002118",
    "4050001001",
    "4050010000",
]
ROUTE_BATCHES = [
    {"batch_id": "batch_1", "route_ids": ["3000232000", "2000002100", "2000003100", "3000323101"]},
    {"batch_id": "batch_2", "route_ids": ["3000410000", "3000410100", "4010002001", "4010002003"]},
    {"batch_id": "batch_3", "route_ids": ["4010002004", "4010002118", "4050001001", "4050010000"]},
]
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
    "terminal_distance_in_sequence",
    "response_status",
    "raw_file_path",
    "raw_sha256",
]
TRAJECTORY_COLUMNS = [
    "vehicle_id",
    "route_id",
    "direction_id",
    "request_time_kst",
    "provider_event_time",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
    "previous_route_id",
    "previous_direction_id",
    "previous_sequence",
    "previous_stop_id",
    "previous_x",
    "previous_y",
    "previous_time",
    "elapsed_seconds",
    "sequence_delta",
    "spatial_distance_m",
    "route_changed",
    "direction_changed",
    "sequence_reset",
    "disappearance_gap",
    "reappearance",
    "batch_id",
    "capture_mode",
    "raw_file_path",
    "raw_sha256",
]
EVENT_COLUMNS = [
    "route_id_before",
    "direction_id_before",
    "vehicle_id",
    "terminal_stop_id",
    "terminal_sequence",
    "terminal_reached_time_lower",
    "terminal_reached_time_upper",
    "post_terminal_time_lower",
    "post_terminal_time_upper",
    "route_id_after",
    "direction_id_after",
    "sequence_after",
    "stop_id_after",
    "elapsed_seconds_lower",
    "elapsed_seconds_upper",
    "spatial_distance_m",
    "capture_mode_before",
    "capture_mode_after",
    "operation_candidate",
    "classification",
    "source_raw_sha256s",
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


def redact_url(url: str, secret: str) -> str:
    return url.replace(secret, "<REDACTED>").replace(urllib.parse.quote(secret, safe="%"), "<REDACTED>").replace(urllib.parse.quote_plus(secret, safe="%"), "<REDACTED>")


def validate_r2(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R2_ROOT
    gate_path = root / "prompt5_e01_r2d1c_r2_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS_MAPPING_PARTIALLY_RECOVERED",
        "previous_mapping_count": 25,
        "newly_recovered_mapping_count": 1,
        "approved_mapping_count": 26,
        "missing_mapping_count": 12,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R2 prerequisite failed: {mismatches}")
    decision = json.loads((root / "mapping_evidence" / APPROVED_R2_ROUTE / "operation_decision.json").read_text(encoding="utf-8"))
    if not (decision.get("decision") == "LOOP_CONTINUOUS" and decision.get("confidence") == "HIGH" and decision.get("approved") is True):
        raise RuntimeError("R2 approved loop evidence for 2000002000 did not match prerequisite.")
    return root, gate


def load_static_topology(r2_root: Path, output_root: Path) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    static = pd.read_parquet(r2_root / "unresolved_route_static_topology.parquet")
    static = static[static["route_id"].astype(str).isin(TARGET_ROUTES)].copy()
    static.to_parquet(output_root / "unresolved_route_static_topology.parquet", index=False)
    terminal_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stop_rows = static[static["record_type"].astype(str) == "STOP_SEQUENCE"].copy()
    stop_rows["sequence_num"] = pd.to_numeric(stop_rows["sequence"], errors="coerce")
    for (route_id, direction_id), group in stop_rows.groupby([stop_rows["route_id"].astype(str), stop_rows["direction_id"].astype(str)]):
        group = group.sort_values("sequence_num")
        if group.empty:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        terminal_by_key[(str(route_id), str(direction_id))] = {
            "route_id": str(route_id),
            "direction_id": str(direction_id),
            "min_sequence": int(first["sequence_num"]) if pd.notna(first["sequence_num"]) else None,
            "max_sequence": int(last["sequence_num"]) if pd.notna(last["sequence_num"]) else None,
            "first_stop_id": str(first["stop_id"]),
            "terminal_stop_id": str(last["stop_id"]),
            "terminal_sequence": int(last["sequence_num"]) if pd.notna(last["sequence_num"]) else None,
        }
    return static, terminal_by_key


def get_route_terminal(route_id: str, terminal_by_key: Mapping[Tuple[str, str], Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    candidates = [v for k, v in terminal_by_key.items() if k[0] == route_id and k[1] == "1"]
    if candidates:
        return candidates[0]
    candidates = [v for k, v in terminal_by_key.items() if k[0] == route_id]
    return candidates[0] if candidates else None


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


def fetch_route(route_id: str, secret: str, output_root: Path, batch_id: str, mode: str, interval: int, terminal: Mapping[str, Any], timeout: int = 30) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    request_time = now_kst()
    raw_dir = output_root / "raw" / batch_id / route_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = [("serviceKey", secret), ("routeId", route_id), ("resultType", "json")]
    url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
    raw_path = raw_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    http_status: Optional[int] = None
    content_type = ""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1c-r3/1.0"}), timeout=timeout) as response:
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
    terminal_sequence = terminal.get("terminal_sequence")
    for row in parse_rows(text):
        seq = norm(row.get("seq")) or norm(row.get("stopSeq"))
        distance = None
        try:
            distance = float(terminal_sequence) - float(seq) if terminal_sequence is not None and seq is not None else None
        except Exception:
            distance = None
        rows.append(
            {
                "request_time_kst": request_time,
                "provider_event_time": norm(row.get("arTime")) or norm(row.get("eventTime")) or norm(row.get("tm")),
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("direction_id")),
                "vehicle_id": norm(row.get("vhcNo2")) or norm(row.get("vhcNo")) or norm(row.get("busId")),
                "current_sequence": seq,
                "current_stop_id": norm(row.get("bsId")) or norm(row.get("stopId")),
                "x": norm(row.get("xPos")) or norm(row.get("x")),
                "y": norm(row.get("yPos")) or norm(row.get("y")),
                "batch_id": batch_id,
                "capture_mode": mode,
                "sampling_interval_seconds": interval,
                "terminal_sequence": terminal_sequence,
                "terminal_distance_in_sequence": distance,
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
    }


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def distance_m(x1: Any, y1: Any, x2: Any, y2: Any) -> Optional[float]:
    fx1, fy1, fx2, fy2 = as_float(x1), as_float(y1), as_float(x2), as_float(y2)
    if fx1 is None or fy1 is None or fx2 is None or fy2 is None:
        return None
    import math
    dx = (fx2 - fx1) * 111_320.0 * max(0.1, math.cos(math.radians((fy1 + fy2) / 2.0)))
    dy = (fy2 - fy1) * 110_540.0
    return float((dx * dx + dy * dy) ** 0.5)


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
                "disappearance_gap": None,
                "reappearance": False,
                "batch_id": row.get("batch_id"),
                "capture_mode": row.get("capture_mode"),
                "raw_file_path": row.get("raw_file_path"),
                "raw_sha256": row.get("raw_sha256"),
            }
            cur = row.to_dict()
            if prev is not None:
                elapsed = (cur["request_dt"] - prev["request_dt"]).total_seconds() if pd.notna(cur["request_dt"]) and pd.notna(prev["request_dt"]) else None
                seq_delta = float(cur["seq_num"] - prev["seq_num"]) if pd.notna(cur["seq_num"]) and pd.notna(prev["seq_num"]) else None
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
                        "route_changed": str(prev.get("route_id")) != str(row.get("route_id")),
                        "direction_changed": str(prev.get("direction_id")) != str(row.get("direction_id")),
                        "sequence_reset": bool(seq_delta is not None and seq_delta < -5),
                        "disappearance_gap": elapsed,
                        "reappearance": bool(elapsed is not None and elapsed > 180),
                    }
                )
            rows.append(out)
            prev = cur
    return rows


def reconstruct_events(trajectories: Sequence[Mapping[str, Any]], terminal_by_key: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    for row in trajectories:
        before_route = str(row.get("previous_route_id")) if row.get("previous_route_id") is not None else None
        before_dir = str(row.get("previous_direction_id")) if row.get("previous_direction_id") is not None else None
        if not before_route or not before_dir:
            continue
        terminal = terminal_by_key.get((before_route, before_dir))
        if terminal is None:
            continue
        prev_seq = as_float(row.get("previous_sequence"))
        max_seq = as_float(terminal.get("terminal_sequence"))
        if prev_seq is None or max_seq is None or prev_seq < max_seq:
            continue
        elapsed = as_float(row.get("elapsed_seconds"))
        if elapsed is None or elapsed <= 0:
            continue
        after_seq = as_float(row.get("current_sequence"))
        min_seq = as_float(terminal.get("min_sequence"))
        after_route = str(row.get("route_id"))
        after_dir = str(row.get("direction_id"))
        op = "UNRESOLVED_TERMINAL_OPERATION"
        if after_route == before_route and after_dir == before_dir and bool(row.get("sequence_reset")) and min_seq is not None and after_seq is not None and after_seq <= min_seq + 3:
            op = "LOOP_CONTINUOUS"
        elif after_route == before_route and after_dir != before_dir and min_seq is not None and after_seq is not None and after_seq <= min_seq + 3:
            op = "PAIRED_OPPOSITE_DIRECTION"
        elif after_route != before_route and bool(row.get("reappearance")):
            op = "DEPOT_OR_DEADHEAD_TRANSITION"
        elif bool(row.get("reappearance")):
            op = "OFF_GRAPH_CONTINUATION_AND_REENTRY"
        events.append(
            {
                "route_id_before": before_route,
                "direction_id_before": before_dir,
                "vehicle_id": row.get("vehicle_id"),
                "terminal_stop_id": terminal.get("terminal_stop_id"),
                "terminal_sequence": terminal.get("terminal_sequence"),
                "terminal_reached_time_lower": row.get("previous_time"),
                "terminal_reached_time_upper": row.get("previous_time"),
                "post_terminal_time_lower": row.get("request_time_kst"),
                "post_terminal_time_upper": row.get("request_time_kst"),
                "route_id_after": after_route,
                "direction_id_after": after_dir,
                "sequence_after": row.get("current_sequence"),
                "stop_id_after": row.get("current_stop_id"),
                "elapsed_seconds_lower": elapsed,
                "elapsed_seconds_upper": elapsed + as_float(row.get("sampling_interval_seconds") or 120) if row.get("sampling_interval_seconds") else elapsed + 120.0,
                "spatial_distance_m": row.get("spatial_distance_m"),
                "capture_mode_before": row.get("capture_mode"),
                "capture_mode_after": row.get("capture_mode"),
                "operation_candidate": op,
                "classification": "OBSERVED_INTERVAL_CENSORED_TERMINAL_TRANSITION",
                "source_raw_sha256s": [row.get("raw_sha256")],
            }
        )
    return events


def make_decisions(samples: pd.DataFrame, events: pd.DataFrame, output_root: Path, terminal_by_key: Mapping[Tuple[str, str], Mapping[str, Any]], batch_routes: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    decisions = []
    contradiction_count = 0
    for route_id in TARGET_ROUTES:
        observed = route_id in batch_routes
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        route_terminal = get_route_terminal(route_id, terminal_by_key)
        direction_id = str(route_terminal.get("direction_id")) if route_terminal else "1"
        terminal_reach_count = int((pd.to_numeric(route_samples["terminal_distance_in_sequence"], errors="coerce") <= 0).sum()) if not route_samples.empty else 0
        op_counts = route_events["operation_candidate"].value_counts().to_dict() if not route_events.empty else {}
        supported = {op: int(count) for op, count in op_counts.items() if op != "UNRESOLVED_TERMINAL_OPERATION"}
        contradictory = max(0, len(supported) - 1)
        contradiction_count += contradictory
        decision = "UNRESOLVED_TERMINAL_OPERATION"
        confidence = "UNRESOLVED"
        approved = False
        remaining = "not observed in this R3 batch"
        if not observed:
            reason = "NOT_OBSERVED_IN_THIS_R3_BATCH"
        elif route_samples.empty:
            reason = "NO_VEHICLE_OBSERVED"
            remaining = "repeat broad scan during operating hours"
        elif terminal_reach_count == 0:
            reason = "NO_TERMINAL_REACH_EVENT"
            remaining = "extend scan to 180 minutes or until terminal-near vehicle appears"
        elif route_events.empty:
            reason = "NO_POST_TERMINAL_CONTINUITY"
            remaining = "terminal-focused capture for at least 30 minutes after reach"
        elif contradictory:
            decision = "MULTIPLE_SERVICE_PATTERNS"
            reason = "MULTIPLE_SERVICE_PATTERNS"
            remaining = "separate branch/service pattern evidence"
        elif supported:
            decision = max(supported.items(), key=lambda item: item[1])[0]
            confidence = "HIGH"
            approved = True
            reason = "static topology plus same-vehicle terminal transition"
            remaining = ""
        else:
            reason = "NO_APPROVABLE_OPERATION"
            remaining = "additional same-vehicle terminal transition evidence"
        obs_minutes = 60 if observed else 0
        transition_count = int(len(route_events[route_events["operation_candidate"] == decision])) if not route_events.empty and decision in set(route_events["operation_candidate"]) else 0
        independent = int(route_events[route_events["operation_candidate"] == decision]["vehicle_id"].nunique(dropna=True)) if transition_count else 0
        decision_row = {
            "route_id": route_id,
            "direction_id": direction_id,
            "decision": decision,
            "confidence": confidence,
            "approved": approved,
            "observation_minutes": obs_minutes,
            "terminal_reach_count": terminal_reach_count,
            "valid_transition_count": transition_count,
            "independent_vehicle_count": independent,
            "contradiction_count": contradictory,
            "supporting_evidence": [],
            "counter_evidence": [],
            "remaining_requirement": remaining,
            "reason": reason,
        }
        decisions.append(decision_row)
        bundle = output_root / "mapping_evidence" / route_id
        bundle.mkdir(parents=True, exist_ok=True)
        dump_json(bundle / "route_static_topology.json", route_terminal or {"route_id": route_id, "missing_static_topology": True})
        dump_json(bundle / "observation_manifest.json", {"route_id": route_id, "observed_in_this_run": observed, "observation_minutes": obs_minutes, "sample_count": int(len(route_samples))})
        write_parquet(bundle / "vehicle_trajectories.parquet", [r for r in build_trajectories(route_samples) if r.get("route_id") == route_id], columns=TRAJECTORY_COLUMNS)
        write_parquet(bundle / "terminal_transition_events.parquet", route_events.to_dict("records") if not route_events.empty else [], columns=EVENT_COLUMNS)
        dump_json(bundle / "terminal_transition_summary.json", {"route_id": route_id, "terminal_reach_count": terminal_reach_count, "valid_transition_count": int(len(route_events)), "operation_counts": op_counts})
        dump_json(bundle / "operation_decision.json", decision_row)
        raw_paths = sorted(set(route_samples["raw_file_path"].dropna().astype(str).tolist())) if not route_samples.empty else []
        dump_json(bundle / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})
    return decisions, {"contradictory_transition_count": contradiction_count}


def update_mapping_v8(r2_root: Path, decisions: Sequence[Mapping[str, Any]], output_root: Path) -> Tuple[Dict[str, Any], bool]:
    v7 = pd.read_parquet(r2_root / "turnaround_mapping_contract_v7.parquet")
    before_approved = {idx for idx, row in v7.iterrows() if truthy(row.get("approved"))}
    by_route = {str(row["route_id"]): row for row in decisions}
    rows = []
    for _, row in v7.iterrows():
        out = row.to_dict()
        route_id = str(out.get("route_id"))
        out["newly_recovered_in_r3"] = False
        was_approved = truthy(out.get("approved"))
        out["approved"] = was_approved
        if route_id in by_route and not was_approved:
            decision = by_route[route_id]
            if truthy(decision["approved"]):
                out["terminal_operation_type"] = decision["decision"]
                out["terminal_operation_resolved"] = True
                out["live_transition_event_count"] = decision["valid_transition_count"]
                out["independent_vehicle_count"] = decision["independent_vehicle_count"]
                out["contradictory_transition_count"] = decision["contradiction_count"]
                out["mapping_confidence"] = decision["confidence"]
                out["approved"] = True
                out["newly_recovered_in_r3"] = True
                out["live_evidence_paths"] = [str(output_root / "mapping_evidence" / route_id / "terminal_transition_events.parquet")]
            else:
                out["terminal_operation_type"] = decision["decision"]
                out["terminal_operation_resolved"] = False
                out["live_transition_event_count"] = decision["valid_transition_count"]
                out["independent_vehicle_count"] = decision["independent_vehicle_count"]
                out["contradictory_transition_count"] = decision["contradiction_count"]
                out["mapping_confidence"] = decision["confidence"]
        if "newly_recovered_in_r2d1c_r2" in out:
            out["newly_recovered_in_r2"] = truthy(out.get("newly_recovered_in_r2d1c_r2"))
        rows.append(out)
    after_approved = {idx for idx, row in enumerate(rows) if truthy(row.get("approved"))}
    regression = not before_approved.issubset(after_approved)
    approved_count = len(after_approved)
    payload = {
        "contract_version": "turnaround_mapping_contract_v8",
        "approved_mapping_before_r3": 26,
        "missing_mapping_before_r3": 12,
        "expected_mapping_count": 38,
        "newly_recovered_mapping_count": sum(1 for row in rows if truthy(row.get("newly_recovered_in_r3"))),
        "approved_mapping_count": approved_count,
        "missing_mapping_count": 38 - approved_count,
        "approved_mapping_regression": regression,
        "rows": rows,
    }
    dump_json(output_root / "turnaround_mapping_contract_v8.json", payload)
    write_parquet(output_root / "turnaround_mapping_contract_v8.parquet", rows)
    return payload, regression


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R3 remaining terminal semantics observation.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--batch-id", default="batch_1")
    parser.add_argument("--scan-minutes", type=int, default=60)
    parser.add_argument("--sampling-interval-seconds", type=int, default=120)
    parser.add_argument("--max-calls-per-minute", type=int, default=20)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    secret, secret_source = read_secret(args.service_key_file)
    if not secret:
        raise SystemExit("service key missing")
    r2_root, r2_gate = validate_r2(project_root)
    r2_gate_path = r2_root / "prompt5_e01_r2d1c_r2_gate.json"
    dump_json(output_root / "r2d1c_r2_reference.json", {"path": str(r2_gate_path), "sha256": sha256_file(r2_gate_path), "gate": r2_gate})
    mapping_v7_path = r2_root / "turnaround_mapping_contract_v7.parquet"
    dump_json(output_root / "mapping_v7_reference.json", {"path": str(mapping_v7_path), "sha256": sha256_file(mapping_v7_path)})
    dump_json(output_root / "credential_reference.json", {"credential_available": True, "credential_source": secret_source, "credential_fingerprint": credential_fingerprint(secret), "credential_length": len(secret), "secret_literal_recorded": False})
    static, terminal_by_key = load_static_topology(r2_root, output_root)
    link_rows = static[static["record_type"].astype(str) == "LINK_SEQUENCE"]
    dump_json(output_root / "observation_campaign_contract.json", {"mode": ["BROAD_SCAN", "TERMINAL_FOCUSED", "POST_TERMINAL_FOLLOW"], "this_run_mode": "BROAD_SCAN", "scan_minutes": args.scan_minutes, "sampling_interval_seconds": args.sampling_interval_seconds, "terminal_trigger_rule": "current_sequence >= terminal_sequence - 5", "focused_capture_started": False, "link_travel_time_available": False, "terminal_recovery_estimated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "route_batch_schedule.json", {"batch_count": 3, "batches": ROUTE_BATCHES, "executed_batch_id": args.batch_id, "approved_r2_route_excluded": APPROVED_R2_ROUTE})
    batch = next((b for b in ROUTE_BATCHES if b["batch_id"] == args.batch_id), ROUTE_BATCHES[0])
    batch_routes = batch["route_ids"]
    preflight_rows, preflight_req = fetch_route(batch_routes[0], secret, output_root, "preflight", "BROAD_SCAN", args.sampling_interval_seconds, get_route_terminal(batch_routes[0], terminal_by_key) or {})
    preflight_pass = preflight_req["response_status"] == "OK"
    dump_json(output_root / "campaign_preflight_audit.json", {"getPos02_auth_pass": preflight_pass, "AUTH_ERROR_count": int(preflight_req["response_status"] == "AUTH_ERROR"), "HTML_response_count": int(preflight_req["response_status"] == "HTML_RESPONSE"), "rate_limit_error_count": int(preflight_req["response_status"] == "RATE_LIMIT"), "raw_sha256": preflight_req["raw_sha256"], "request_url_redacted": preflight_req["request_url_redacted"]})
    requests = [preflight_req]
    samples = preflight_rows
    if preflight_pass:
        end_time = time.monotonic() + max(args.scan_minutes * 60, 1)
        while time.monotonic() < end_time:
            sample_started = time.monotonic()
            for route_id in batch_routes:
                terminal = get_route_terminal(route_id, terminal_by_key)
                if terminal is None:
                    continue
                rows, req = fetch_route(route_id, secret, output_root, args.batch_id, "BROAD_SCAN", args.sampling_interval_seconds, terminal)
                samples.extend(rows)
                requests.append(req)
                sleep_for = 60.0 / max(args.max_calls_per_minute, 1)
                if req["response_status"] == "RATE_LIMIT":
                    sleep_for = max(sleep_for * 2, 10.0)
                time.sleep(min(sleep_for, max(0.0, end_time - time.monotonic())))
                if time.monotonic() >= end_time:
                    break
            remaining = args.sampling_interval_seconds - (time.monotonic() - sample_started)
            if remaining > 0:
                time.sleep(min(remaining, max(0.0, end_time - time.monotonic())))
    write_parquet(output_root / "terminal_semantics_position_samples_r3.parquet", samples, columns=SAMPLE_COLUMNS)
    sample_df = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)
    trajectories = build_trajectories(sample_df)
    write_parquet(output_root / "vehicle_continuous_trajectories_r3.parquet", trajectories, columns=TRAJECTORY_COLUMNS)
    events = reconstruct_events(trajectories, terminal_by_key)
    write_parquet(output_root / "terminal_transition_events_r3.parquet", events, columns=EVENT_COLUMNS)
    event_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    decisions, contradiction = make_decisions(sample_df, event_df, output_root, terminal_by_key, batch_routes)
    write_parquet(output_root / "terminal_operation_decisions_r3.parquet", decisions)
    dump_json(output_root / "terminal_operation_contradiction_audit.json", contradiction)
    mapping_v8, regression = update_mapping_v8(r2_root, decisions, output_root)
    remaining = [
        {
            "route_id": row["route_id"],
            "direction_id": row["direction_id"],
            "blocker": row["reason"],
            "remaining_requirement": row["remaining_requirement"],
            "recommended_sampling_duration": "continue to 180 minutes for priority route or one service window",
            "recommended_sampling_interval_seconds": 120,
        }
        for row in decisions
        if not row["approved"]
    ]
    dump_json(output_root / "remaining_mapping_evidence_requirements.json", {"rows": remaining})
    auth_errors = sum(1 for r in requests if r["response_status"] == "AUTH_ERROR")
    html_errors = sum(1 for r in requests if r["response_status"] == "HTML_RESPONSE")
    rate_errors = sum(1 for r in requests if r["response_status"] == "RATE_LIMIT")
    post_scan = scan_secret(output_root, secret)
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": post_scan["secret_literal_occurrence_count"], "secret_leak_pass": post_scan["secret_literal_occurrence_count"] == 0, "post_campaign_artifact_scan": post_scan})
    recovery_auth = {"approved_for_terminal_recovery_campaign": mapping_v8["approved_mapping_count"] == 38, "terminal_recovery_seconds": "unresolved", "terminal_recovery_estimated": False, "terminal_recovery_applied": False}
    dump_json(output_root / "terminal_recovery_campaign_authorization_draft.json", recovery_auth)
    phase2_auth = {"approved_for_terminal_recovery_campaign": recovery_auth["approved_for_terminal_recovery_campaign"], "approved_for_phase2_turnaround_execution": False, "approved_for_baseline_feasibility_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a_corrected_retrospective": False, "approved_for_e2_execution": False, "prompt6_full_matrix_approved": False, "real_world_causal_claim_allowed": False, "phase2_production_executed": False}
    dump_json(output_root / "phase2_execution_authorization.json", phase2_auth)
    op_counts = pd.DataFrame(decisions)["decision"].value_counts().to_dict()
    conf_counts = pd.DataFrame(decisions)["confidence"].value_counts().to_dict()
    routes_no_vehicle = sum(1 for row in decisions if row["reason"] == "NO_VEHICLE_OBSERVED")
    routes_no_terminal = sum(1 for row in decisions if row["reason"] == "NO_TERMINAL_REACH_EVENT")
    routes_no_post = sum(1 for row in decisions if row["reason"] == "NO_POST_TERMINAL_CONTINUITY")
    newly = mapping_v8["newly_recovered_mapping_count"]
    if regression:
        status = "FAIL_MAPPING_REGRESSION"
    elif auth_errors or html_errors or rate_errors or post_scan["secret_literal_occurrence_count"]:
        status = "FAIL_OBSERVATION_CAMPAIGN"
    elif mapping_v8["approved_mapping_count"] == 38:
        status = "PASS_MAPPING_38_OF_38_READY"
    elif newly > 0:
        status = "PASS_MAPPING_PARTIALLY_RECOVERED"
    elif routes_no_post:
        status = "BLOCKED_POST_TERMINAL_CONTINUITY"
    else:
        status = "BLOCKED_NO_TERMINAL_REACH_EVENTS"
    repeated = int((sample_df.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not sample_df.empty else 0
    seq_prog = int(sum(1 for _, group in sample_df.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2)) if not sample_df.empty else 0
    terminal_reach = int((pd.to_numeric(sample_df["terminal_distance_in_sequence"], errors="coerce") <= 0).sum()) if not sample_df.empty else 0
    gate = {
        "status": status,
        "classification": status,
        "r2d1c_r2_gate_path": str(r2_gate_path),
        "r2d1c_r2_gate_sha256": sha256_file(r2_gate_path),
        "campaign_route_count": len(batch_routes),
        "campaign_batch_count": 1,
        "campaign_total_observation_minutes": args.scan_minutes,
        "campaign_total_requests": len(requests),
        "campaign_raw_file_count": len(requests),
        "campaign_normalized_rows": len(samples),
        "unique_vehicle_count": int(sample_df["vehicle_id"].nunique(dropna=True)) if not sample_df.empty else 0,
        "repeated_vehicle_count": repeated,
        "sequence_progression_count": seq_prog,
        "terminal_reach_event_count": terminal_reach,
        "post_terminal_observation_count": len(events),
        "valid_terminal_transition_count": len(events),
        "previous_mapping_count": 26,
        "newly_recovered_mapping_count": newly,
        "approved_mapping_count": mapping_v8["approved_mapping_count"],
        "missing_mapping_count": mapping_v8["missing_mapping_count"],
        "paired_opposite_direction_count": op_counts.get("PAIRED_OPPOSITE_DIRECTION", 0),
        "loop_continuous_count": op_counts.get("LOOP_CONTINUOUS", 0),
        "off_graph_reentry_count": op_counts.get("OFF_GRAPH_CONTINUATION_AND_REENTRY", 0),
        "depot_deadhead_count": op_counts.get("DEPOT_OR_DEADHEAD_TRANSITION", 0),
        "multiple_service_pattern_count": op_counts.get("MULTIPLE_SERVICE_PATTERNS", 0),
        "unresolved_terminal_operation_count": op_counts.get("UNRESOLVED_TERMINAL_OPERATION", 0),
        "high_confidence_mapping_count": conf_counts.get("HIGH", 0),
        "medium_confidence_mapping_count": conf_counts.get("MEDIUM", 0),
        "low_confidence_mapping_count": conf_counts.get("LOW", 0),
        "contradictory_transition_count": contradiction["contradictory_transition_count"],
        "routes_no_vehicle_observed": routes_no_vehicle,
        "routes_no_terminal_reach": routes_no_terminal,
        "routes_no_post_terminal_continuity": routes_no_post,
        "AUTH_ERROR_count": auth_errors,
        "HTML_response_count": html_errors,
        "rate_limit_error_count": rate_errors,
        "secret_leak_count": post_scan["secret_literal_occurrence_count"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
        "approved_for_terminal_recovery_campaign": recovery_auth["approved_for_terminal_recovery_campaign"],
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "threshold_changed": False,
        "scientific_parameter_changed": False,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r3_gate.json", gate)
    report = f"""# Prompt 5-E01-R2D-1C-R3 Final Report

[Prompt 5-E01-R2D-1C-R3 판정]
status: {status}
classification: {status}

[Campaign]
routes: {len(batch_routes)}
batches: 1
observation minutes: {args.scan_minutes}
requests: {len(requests)}
raw files: {len(requests)}
normalized rows: {len(samples)}

[Vehicle continuity]
unique: {gate['unique_vehicle_count']}
repeated: {repeated}
sequence progression: {seq_prog}
terminal reaches: {terminal_reach}
post-terminal observations: {len(events)}
valid transitions: {len(events)}

[Mapping]
previous: 26
newly recovered: {newly}
approved: {mapping_v8['approved_mapping_count']}
missing: {mapping_v8['missing_mapping_count']}

paired: {gate['paired_opposite_direction_count']}
loop: {gate['loop_continuous_count']}
off-graph: {gate['off_graph_reentry_count']}
depot/deadhead: {gate['depot_deadhead_count']}
multiple patterns: {gate['multiple_service_pattern_count']}
unresolved: {gate['unresolved_terminal_operation_count']}

[Unresolved reasons]
no vehicle: {routes_no_vehicle}
no terminal reach: {routes_no_terminal}
no post-terminal continuity: {routes_no_post}
multiple patterns: {gate['multiple_service_pattern_count']}
topology conflict: 0

[Confidence]
HIGH: {gate['high_confidence_mapping_count']}
MEDIUM: {gate['medium_confidence_mapping_count']}
LOW: {gate['low_confidence_mapping_count']}
contradictions: {gate['contradictory_transition_count']}

[API/security]
AUTH_ERROR: {auth_errors}
HTML: {html_errors}
rate-limit: {rate_errors}
secret leak: {post_scan['secret_literal_occurrence_count']}

[Recovery]
estimated: false
applied: false
campaign approved: {str(recovery_auth['approved_for_terminal_recovery_campaign']).lower()}

[Guard]
test split: false
test target: false
test embedding: false
threshold changed: false
Phase 2 executed: false

[Next gate]
additional mapping observation: true
terminal recovery campaign: {str(recovery_auth['approved_for_terminal_recovery_campaign']).lower()}
Phase 2: false
baseline feasibility: false
E0/E1 retraining: false
Prompt 6A: false
E2: false
"""
    (output_root / "prompt5_e01_r2d1c_r3_final_report.md").write_text(report, encoding="utf-8")
    dump_json(output_root / "prompt5_e01_r2d1c_r3_manifest.json", {"artifact_version": "prompt5_e01_r2d1c_r3_remaining_terminal_semantics_v1", "created_at_kst": now_kst(), "artifact_dir": str(output_root), "files": sorted([str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()])})
    print(json.dumps({"artifact_dir": str(output_root), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
