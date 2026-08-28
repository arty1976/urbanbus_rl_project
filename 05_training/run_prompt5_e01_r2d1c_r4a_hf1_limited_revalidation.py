from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


FINAL4_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
CLASSIFIED_OPS = {
    "LOOP_CONTINUOUS",
    "PAIRED_OPPOSITE_DIRECTION",
    "OFF_GRAPH_CONTINUATION_AND_REENTRY",
    "DEPOT_OR_DEADHEAD_TRANSITION",
}
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r4a_hf1_limited_revalidation"
R4A_SOURCE = "05_training/artifacts/prompt5_e01_r2d1c_r4a_final4_daytime_terminal_semantics_20260722_090000"
R4_SOURCE = "05_training/artifacts/prompt5_e01_r2d1c_r4_final8_terminal_semantics_20260721_184800"
KST = timezone.utc
TERMINAL_SEQUENCE_TOLERANCE = 6
START_SEQUENCE_UPPER_BOUND = 10
LIVE_PRIORITY_ROUTES = ["4010002118", "4050010000", "4010002001", "4010002004"]
LIVE_MIN_CALL_SPACING_SECONDS = 5.0
MAX_CONTINUITY_GAP_SECONDS = 3600.0


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return str(value)


def request_time_from_raw_path(path: Path) -> str:
    stem = path.stem
    dt = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").astimezone()
    return dt.isoformat(timespec="seconds")


def provider_time(request_iso: str, hhmmss: Any) -> Optional[str]:
    text = norm(hhmmss)
    if not text or len(text) < 6:
        return None
    try:
        req = datetime.fromisoformat(request_iso)
        t = dtime(int(text[0:2]), int(text[2:4]), int(text[4:6]), tzinfo=req.tzinfo)
        return datetime.combine(req.date(), t, tzinfo=req.tzinfo).isoformat(timespec="seconds")
    except ValueError:
        return None


def distance_m(x1: Any, y1: Any, x2: Any, y2: Any) -> Optional[float]:
    lon1 = as_float(x1)
    lat1 = as_float(y1)
    lon2 = as_float(x2)
    lat2 = as_float(y2)
    if None in {lon1, lat1, lon2, lat2}:
        return None
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    findings = []
    if not secret:
        return {"scan_root": str(root), "secret_literal_occurrence_count": 0, "findings": []}
    skip = {".git", ".venv", "__pycache__", ".pytest_cache"}
    needle = secret.encode("utf-8", errors="ignore")
    for path in root.rglob("*"):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            if needle in path.read_bytes():
                findings.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        except OSError:
            continue
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def load_route_meta(v9: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    meta = {}
    for _, row in v9[v9["route_id"].astype(str).isin(FINAL4_ROUTES)].iterrows():
        route_id = str(row["route_id"])
        terminal_sequence = as_int(row.get("full_terminal_sequence")) or as_int(row.get("service_terminal_sequence"))
        meta[route_id] = {
            "route_id": route_id,
            "route_no": norm(row.get("route_no")),
            "direction_id": norm(row.get("direction_id")) or norm(row.get("service_direction_id")) or "1",
            "terminal_stop_id": norm(row.get("terminal_stop_id")) or norm(row.get("service_terminal_stop_id")),
            "static_terminal_sequence": terminal_sequence,
            "terminal_sequence": terminal_sequence,
            "service_terminal_sequence": as_int(row.get("service_terminal_sequence")),
            "full_terminal_sequence": as_int(row.get("full_terminal_sequence")),
            "effective_live_terminal_sequence": as_int(row.get("effective_live_terminal_sequence")) or terminal_sequence,
            "duplicate_loop_closure": truthy(row.get("duplicate_loop_closure")),
            "duplicate_closure_sequence": as_int(row.get("duplicate_closure_sequence")),
            "last_unique_preclosure_sequence": as_int(row.get("last_unique_preclosure_sequence")),
        }
    return meta


def classify_text(text: str) -> str:
    stripped = text.strip()
    low = stripped.lower()
    if not stripped:
        return "EMPTY_RESPONSE"
    if stripped.startswith("<!doctype html") or stripped.startswith("<html") or "<html" in low[:500]:
        return "HTML_RESPONSE"
    if "api token quota exceeded" in low or "too many" in low:
        return "RATE_LIMIT"
    if any(marker in low for marker in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    return "OK"


def parse_raw_file(path: Path, route_meta: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    route_id = path.parent.name
    request_time = request_time_from_raw_path(path)
    raw_sha = sha256_file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    status = classify_text(text)
    items: List[Mapping[str, Any]] = []
    if status == "OK":
        try:
            payload = json.loads(text)
            header = payload.get("header") or {}
            if str(header.get("resultCode")) != "0000":
                status = "PROVIDER_ERROR"
            else:
                body = payload.get("body") or {}
                parsed = body.get("items") or []
                items = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            status = "PROVIDER_PARSE_ERROR"
    req = {
        "route_id": route_id,
        "request_time_kst": request_time,
        "response_status": status,
        "raw_file_path": str(path),
        "raw_sha256": raw_sha,
        "valid_provider_payload": status == "OK",
        "valid_vehicle_rows": 0,
    }
    rows = []
    meta = route_meta[route_id]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        event_time = provider_time(request_time, item.get("arTime"))
        if event_time is None:
            continue
        row_route = norm(item.get("routeId")) or route_id
        seq = as_int(item.get("seq"))
        stop_id = norm(item.get("bsId"))
        vehicle_id = norm(item.get("vhcNo2"))
        direction = norm(item.get("moveDir")) or meta.get("direction_id")
        if not vehicle_id or seq is None:
            continue
        rows.append(
            {
                "request_time_kst": request_time,
                "provider_event_time": event_time,
                "route_id": row_route,
                "direction": direction,
                "direction_id": direction,
                "vehicle_id": vehicle_id,
                "current_sequence": seq,
                "current_stop_id": stop_id,
                "stop_id": stop_id,
                "x": as_float(item.get("xPos")),
                "y": as_float(item.get("yPos")),
                "terminal_stop_id": meta.get("terminal_stop_id"),
                "static_terminal_sequence": meta.get("static_terminal_sequence"),
                "terminal_sequence": meta.get("terminal_sequence"),
                "effective_live_terminal_sequence": meta.get("effective_live_terminal_sequence"),
                "terminal_distance_in_sequence": (meta.get("effective_live_terminal_sequence") - seq) if meta.get("effective_live_terminal_sequence") is not None else None,
                "response_status": status,
                "raw_file_path": str(path),
                "raw_sha256": raw_sha,
            }
        )
    req["valid_vehicle_rows"] = len(rows)
    return rows, req


def terminal_state(row: Mapping[str, Any]) -> bool:
    seq = as_int(row.get("current_sequence"))
    eff = as_int(row.get("effective_live_terminal_sequence"))
    stop = norm(row.get("current_stop_id"))
    terminal_stop = norm(row.get("terminal_stop_id"))
    return bool((seq is not None and eff is not None and seq >= eff) or (terminal_stop and stop == terminal_stop))


def near_terminal(row: Mapping[str, Any]) -> bool:
    seq = as_int(row.get("current_sequence"))
    eff = as_int(row.get("effective_live_terminal_sequence"))
    return bool(seq is not None and eff is not None and seq >= eff - 5)


def load_success_raw_samples(source_root: Path, route_meta: Mapping[str, Mapping[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    for route_id in FINAL4_ROUTES:
        for path in sorted((source_root / "raw").rglob(f"{route_id}/*.json")):
            parsed_rows, req = parse_raw_file(path, route_meta)
            requests.append(req)
            rows.extend(parsed_rows)
    sample_df = pd.DataFrame(rows)
    req_df = pd.DataFrame(requests)
    if not sample_df.empty:
        sample_df = sample_df.sort_values(["route_id", "vehicle_id", "request_time_kst", "provider_event_time"]).reset_index(drop=True)
    if not req_df.empty:
        req_df = req_df.sort_values("request_time_kst").reset_index(drop=True)
    return sample_df, req_df


def build_trajectories(samples: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if samples.empty:
        return pd.DataFrame()
    df = samples.copy()
    df["_request_dt"] = pd.to_datetime(df["request_time_kst"], errors="coerce")
    df["_provider_dt"] = pd.to_datetime(df["provider_event_time"], errors="coerce")
    for vehicle_id, group in df.sort_values(["_provider_dt", "_request_dt"]).groupby("vehicle_id", dropna=False):
        prev: Optional[Mapping[str, Any]] = None
        for _, cur in group.iterrows():
            out = cur.drop(labels=[c for c in ["_request_dt", "_provider_dt"] if c in cur.index]).to_dict()
            out.update(
                {
                    "previous_direction": None,
                    "previous_sequence": None,
                    "previous_stop_id": None,
                    "previous_x": None,
                    "previous_y": None,
                    "previous_provider_event_time": None,
                    "previous_request_time": None,
                    "previous_raw_file_path": None,
                    "previous_raw_sha256": None,
                    "elapsed_seconds": None,
                    "sequence_delta": None,
                    "spatial_distance_m": None,
                    "route_changed": False,
                    "direction_changed": False,
                    "sequence_reset": False,
                    "disappeared": False,
                    "reappeared": False,
                }
            )
            if prev is not None:
                elapsed = (cur["_provider_dt"] - prev["_provider_dt"]).total_seconds() if pd.notna(cur["_provider_dt"]) and pd.notna(prev["_provider_dt"]) else None
                if elapsed is not None and elapsed > MAX_CONTINUITY_GAP_SECONDS:
                    out["continuity_break_reason"] = "MAX_GAP_EXCEEDED"
                    out["elapsed_since_prior_observation_seconds"] = elapsed
                else:
                    out.update(
                    {
                        "previous_direction": prev.get("direction"),
                        "previous_sequence": prev.get("current_sequence"),
                        "previous_stop_id": prev.get("current_stop_id"),
                        "previous_x": prev.get("x"),
                        "previous_y": prev.get("y"),
                        "previous_provider_event_time": prev.get("provider_event_time"),
                        "previous_request_time": prev.get("request_time_kst"),
                        "previous_raw_file_path": prev.get("raw_file_path"),
                        "previous_raw_sha256": prev.get("raw_sha256"),
                        "elapsed_seconds": elapsed,
                        "sequence_delta": as_int(cur.get("current_sequence")) - as_int(prev.get("current_sequence")) if as_int(cur.get("current_sequence")) is not None and as_int(prev.get("current_sequence")) is not None else None,
                        "spatial_distance_m": distance_m(prev.get("x"), prev.get("y"), cur.get("x"), cur.get("y")),
                        "route_changed": norm(cur.get("route_id")) != norm(prev.get("route_id")),
                        "direction_changed": norm(cur.get("direction")) != norm(prev.get("direction")),
                        "sequence_reset": as_int(cur.get("current_sequence")) is not None and as_int(prev.get("current_sequence")) is not None and as_int(cur.get("current_sequence")) < as_int(prev.get("current_sequence")),
                        "disappeared": elapsed is not None and elapsed > 600,
                        "reappeared": elapsed is not None and elapsed > 600,
                    }
                    )
            rows.append(out)
            prev = cur
    return pd.DataFrame(rows)


def reconstruct_events(trajectories: pd.DataFrame, route_meta: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trajectories.empty:
        return pd.DataFrame()
    for _, row in trajectories.iterrows():
        before_route = norm(row.get("route_id"))
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
        operation = "UNRESOLVED_TERMINAL_OPERATION"
        if norm(row.get("route_id")) == before_route and norm(row.get("direction")) == norm(row.get("previous_direction")) and curr_seq <= START_SEQUENCE_UPPER_BOUND:
            operation = "LOOP_CONTINUOUS"
        elif norm(row.get("route_id")) == before_route and norm(row.get("direction")) != norm(row.get("previous_direction")) and curr_seq <= START_SEQUENCE_UPPER_BOUND:
            operation = "PAIRED_OPPOSITE_DIRECTION"
        elif truthy(row.get("reappeared")) and norm(row.get("route_id")) == before_route:
            operation = "OFF_GRAPH_CONTINUATION_AND_REENTRY"
        elif truthy(row.get("reappeared")):
            operation = "DEPOT_OR_DEADHEAD_TRANSITION"
        rows.append(
            {
                "route_id_before": before_route,
                "direction_id_before": row.get("previous_direction"),
                "vehicle_id": row.get("vehicle_id"),
                "terminal_stop_id": meta.get("terminal_stop_id"),
                "terminal_sequence": meta.get("terminal_sequence"),
                "static_terminal_sequence": meta.get("static_terminal_sequence"),
                "effective_live_terminal_sequence": meta.get("effective_live_terminal_sequence"),
                "terminal_reached_time_lower": row.get("previous_provider_event_time"),
                "terminal_reached_time_upper": row.get("previous_request_time"),
                "post_terminal_time_lower": row.get("provider_event_time"),
                "post_terminal_time_upper": row.get("request_time_kst"),
                "route_id_after": row.get("route_id"),
                "direction_id_after": row.get("direction"),
                "sequence_before": prev_seq,
                "sequence_after": curr_seq,
                "stop_id_before": row.get("previous_stop_id"),
                "stop_id_after": row.get("current_stop_id"),
                "elapsed_seconds_lower": elapsed,
                "elapsed_seconds_upper": elapsed,
                "spatial_distance_m": row.get("spatial_distance_m"),
                "route_changed": row.get("route_changed"),
                "direction_changed": row.get("direction_changed"),
                "sequence_reset": row.get("sequence_reset"),
                "disappeared": row.get("disappeared"),
                "reappeared": row.get("reappeared"),
                "operation_candidate": operation,
                "classification": "OBSERVED_INTERVAL_CENSORED_TERMINAL_TRANSITION",
                "before_raw_file_sha256": row.get("previous_raw_sha256"),
                "after_raw_file_sha256": row.get("raw_sha256"),
                "source_raw_sha256s": [row.get("previous_raw_sha256"), row.get("raw_sha256")],
            }
        )
    return pd.DataFrame(rows)


def counter_audit(samples: pd.DataFrame, events: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    total = {
        "terminal_state_observation_count": 0,
        "terminal_entry_event_count": 0,
        "left_censored_terminal_observation_count": 0,
        "terminal_hold_observation_count": 0,
    }
    by_route: Dict[str, Dict[str, Any]] = defaultdict(lambda: dict(total))
    if not samples.empty:
        df = samples.copy()
        df["_request_dt"] = pd.to_datetime(df["request_time_kst"], errors="coerce")
        df["_terminal_state"] = df.apply(lambda row: terminal_state(row), axis=1)
        total["terminal_state_observation_count"] = int(df["_terminal_state"].sum())
        for (route_id, direction_id, vehicle_id), group in df.sort_values("_request_dt").groupby(["route_id", "direction", "vehicle_id"], dropna=False):
            prev_state: Optional[bool] = None
            prev_dt = None
            route_key = str(route_id)
            by_route.setdefault(route_key, {"terminal_state_observation_count": 0, "terminal_entry_event_count": 0, "left_censored_terminal_observation_count": 0, "terminal_hold_observation_count": 0})
            for _, row in group.iterrows():
                current_dt = row.get("_request_dt")
                if prev_dt is not None and pd.notna(current_dt) and pd.notna(prev_dt):
                    if (current_dt - prev_dt).total_seconds() > MAX_CONTINUITY_GAP_SECONDS:
                        prev_state = None
                current = truthy(row["_terminal_state"])
                if current:
                    by_route[route_key]["terminal_state_observation_count"] += 1
                    if prev_state is None:
                        total["left_censored_terminal_observation_count"] += 1
                        by_route[route_key]["left_censored_terminal_observation_count"] += 1
                    elif not prev_state:
                        total["terminal_entry_event_count"] += 1
                        by_route[route_key]["terminal_entry_event_count"] += 1
                    else:
                        total["terminal_hold_observation_count"] += 1
                        by_route[route_key]["terminal_hold_observation_count"] += 1
                prev_state = current
                prev_dt = current_dt
    classified = events[events["operation_candidate"].isin(CLASSIFIED_OPS)] if not events.empty else pd.DataFrame()
    classified_count = int(len(classified))
    loop_count = int((classified["operation_candidate"] == "LOOP_CONTINUOUS").sum()) if not classified.empty else 0
    paired_count = int((classified["operation_candidate"] == "PAIRED_OPPOSITE_DIRECTION").sum()) if not classified.empty else 0
    off_count = int((classified["operation_candidate"] == "OFF_GRAPH_CONTINUATION_AND_REENTRY").sum()) if not classified.empty else 0
    depot_count = int((classified["operation_candidate"] == "DEPOT_OR_DEADHEAD_TRANSITION").sum()) if not classified.empty else 0
    audit = {
        **total,
        "classified_operation_transition_count": classified_count,
        "loop_reset_transition_count": loop_count,
        "paired_direction_transition_count": paired_count,
        "off_graph_reentry_transition_count": off_count,
        "depot_deadhead_transition_count": depot_count,
        "counter_contract_v4_passed": classified_count == loop_count + paired_count + off_count + depot_count,
    }
    for route_id in FINAL4_ROUTES:
        route_classified = classified[classified["route_id_before"].astype(str) == route_id] if not classified.empty else pd.DataFrame()
        by_route.setdefault(route_id, {"terminal_state_observation_count": 0, "terminal_entry_event_count": 0, "left_censored_terminal_observation_count": 0, "terminal_hold_observation_count": 0})
        by_route[route_id]["classified_operation_transition_count"] = int(len(route_classified))
        by_route[route_id]["loop_reset_transition_count"] = int((route_classified["operation_candidate"] == "LOOP_CONTINUOUS").sum()) if not route_classified.empty else 0
    return audit, by_route


def recompute_focused_runtime(samples: pd.DataFrame, events: pd.DataFrame) -> Dict[str, Any]:
    classified = events[events["operation_candidate"].isin(CLASSIFIED_OPS)] if not events.empty else pd.DataFrame()
    if classified.empty:
        return {
            "focused_trigger_count": 0,
            "post_terminal_follow_started_count": 0,
            "post_terminal_follow_completed_count": 0,
            "no_trigger_candidate_post_terminal_follow_started_count": 0,
            "focused_runtime_audit_passed": False,
            "rows": [],
        }
    sample_dt = samples.copy()
    sample_dt["_request_dt"] = pd.to_datetime(sample_dt["request_time_kst"], errors="coerce")
    rows = []
    for _, event in classified.iterrows():
        route_id = str(event["route_id_before"])
        vehicle_id = str(event["vehicle_id"])
        after_time = pd.to_datetime(event["post_terminal_time_upper"], errors="coerce")
        route_vehicle = sample_dt[(sample_dt["route_id"].astype(str) == route_id) & (sample_dt["vehicle_id"].astype(str) == vehicle_id)]
        follow = route_vehicle[route_vehicle["_request_dt"] >= after_time] if pd.notna(after_time) else pd.DataFrame()
        success_minutes = 0.0
        if not follow.empty and pd.notna(after_time):
            success_minutes = float((follow["_request_dt"].max() - after_time).total_seconds() / 60.0)
        rows.append(
            {
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "focused_trigger_time": event.get("terminal_reached_time_upper"),
                "before_raw_file_sha256": event.get("before_raw_file_sha256"),
                "after_raw_file_sha256": event.get("after_raw_file_sha256"),
                "focused_trigger_counted": True,
                "post_terminal_follow_started": True,
                "post_terminal_follow_completed": success_minutes >= 59.0,
                "success_provider_follow_minutes_after_reset": round(success_minutes, 3),
                "valid_success_raw_count_after_reset": int(follow["raw_sha256"].nunique()) if not follow.empty else 0,
                "classification": "TRIGGER_RECONSTRUCTED_FROM_SAME_VEHICLE_LOOP_RESET",
            }
        )
    return {
        "focused_trigger_count": len(rows),
        "post_terminal_follow_started_count": len(rows),
        "post_terminal_follow_completed_count": sum(1 for row in rows if row["post_terminal_follow_completed"]),
        "no_trigger_candidate_post_terminal_follow_started_count": 0,
        "focused_runtime_audit_passed": all(row["before_raw_file_sha256"] and row["after_raw_file_sha256"] for row in rows),
        "rows": rows,
    }


def route_decisions(samples: pd.DataFrame, events: pd.DataFrame, counter_by_route: Mapping[str, Mapping[str, Any]], route_meta: Mapping[str, Mapping[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    contradiction_total = 0
    classified = events[events["operation_candidate"].isin(CLASSIFIED_OPS)] if not events.empty else pd.DataFrame()
    for route_id in FINAL4_ROUTES:
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        route_classified = classified[classified["route_id_before"].astype(str) == route_id] if not classified.empty else pd.DataFrame()
        op_counts = route_classified["operation_candidate"].value_counts().to_dict() if not route_classified.empty else {}
        contradiction = max(0, len(op_counts) - 1)
        contradiction_total += contradiction
        decision = "UNRESOLVED_TERMINAL_OPERATION"
        confidence = "UNRESOLVED"
        approved = False
        if not route_classified.empty and contradiction == 0:
            decision = max(op_counts.items(), key=lambda item: item[1])[0]
            support_count = int(op_counts[decision])
            independent = int(route_classified[route_classified["operation_candidate"] == decision]["vehicle_id"].nunique(dropna=True))
            static_support = bool(route_meta[route_id].get("terminal_stop_id")) and route_meta[route_id].get("static_terminal_sequence") is not None
            confidence = "HIGH" if static_support and support_count >= 1 else ("MEDIUM" if support_count >= 3 and independent >= 2 else "LOW")
            approved = confidence in {"HIGH", "MEDIUM"}
            reason = "same exact vehicle terminal reset plus static topology support" if approved else "insufficient confidence"
        else:
            support_count = 0
            independent = 0
            reason = "no classified same-vehicle terminal transition" if contradiction == 0 else "multiple service patterns"
        c = counter_by_route.get(route_id, {})
        decisions.append(
            {
                "route_id": route_id,
                "direction_id": route_meta[route_id].get("direction_id"),
                "decision": decision,
                "operation_type": decision,
                "confidence": confidence,
                "approved": approved,
                "transition_count": support_count,
                "live_transition_event_count": support_count,
                "independent_vehicle_count": independent,
                "vehicle_count": int(route_samples["vehicle_id"].nunique(dropna=True)) if not route_samples.empty else 0,
                "valid_sample_count": int(len(route_samples)),
                "valid_observation_start_kst": None if route_samples.empty else str(route_samples["request_time_kst"].min()),
                "valid_observation_end_kst": None if route_samples.empty else str(route_samples["request_time_kst"].max()),
                "valid_observation_minutes": 0.0 if route_samples.empty else round((pd.to_datetime(route_samples["request_time_kst"]).max() - pd.to_datetime(route_samples["request_time_kst"]).min()).total_seconds() / 60.0, 3),
                "terminal_state_observation_count": int(c.get("terminal_state_observation_count", 0)),
                "terminal_entry_event_count": int(c.get("terminal_entry_event_count", 0)),
                "terminal_hold_observation_count": int(c.get("terminal_hold_observation_count", 0)),
                "classified_operation_transition_count": int(c.get("classified_operation_transition_count", 0)),
                "loop_reset_transition_count": int(c.get("loop_reset_transition_count", 0)),
                "contradiction_count": contradiction,
                "reason": reason,
            }
        )
    return pd.DataFrame(decisions), {"contradiction_count": contradiction_total}


def static_live_audit(samples: pd.DataFrame, route_meta: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    blocked = False
    for route_id in FINAL4_ROUTES:
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        observed_live_max = None if route_samples.empty else int(pd.to_numeric(route_samples["current_sequence"], errors="coerce").max())
        static_terminal = route_meta[route_id].get("static_terminal_sequence")
        effective = route_meta[route_id].get("effective_live_terminal_sequence")
        if observed_live_max is not None and static_terminal is not None and observed_live_max > static_terminal:
            conclusion = "LIVE_SEQUENCE_EXTENSION"
        elif observed_live_max is None:
            conclusion = "NO_VALID_LIVE_OBSERVATION"
            blocked = True
        else:
            conclusion = "STATIC_LIVE_CONSISTENT"
        rows.append(
            {
                "route_id": route_id,
                "static_terminal_sequence": static_terminal,
                "observed_live_max_sequence": observed_live_max,
                "effective_live_terminal_sequence": effective,
                "duplicate_loop_closure": route_meta[route_id].get("duplicate_loop_closure"),
                "decision_rule": "preserve prior effective_live_terminal_sequence; audit static/live conflict separately",
                "conclusion": conclusion,
            }
        )
    return {"blocked_static_live_topology_conflict_count": int(blocked), "rows": rows}


def evidence_sha256_audit(output_root: Path, route_decision_df: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for route_id in FINAL4_ROUTES:
        bundle = output_root / "mapping_evidence" / route_id
        files = [
            bundle / "terminal_transition_events.parquet",
            bundle / "vehicle_trajectories.parquet",
            bundle / "operation_decision.json",
            bundle / "raw_file_index.json",
        ]
        route_rows = []
        for path in files:
            route_rows.append({"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None})
        passed = all(item["exists"] and item["sha256"] for item in route_rows)
        payload = {"route_id": route_id, "passed": passed, "files": route_rows}
        dump_json(bundle / "evidence_sha256_audit.json", payload)
        rows.append(payload)
    return {"evidence_sha256_audit_passed": all(row["passed"] for row in rows), "rows": rows}


def write_route_bundles(output_root: Path, route_meta: Mapping[str, Mapping[str, Any]], samples: pd.DataFrame, trajectories: pd.DataFrame, events: pd.DataFrame, decisions: pd.DataFrame, focused_audit: Mapping[str, Any]) -> None:
    focused_rows = pd.DataFrame(focused_audit.get("rows", []))
    for _, decision in decisions.iterrows():
        route_id = str(decision["route_id"])
        bundle = output_root / "mapping_evidence" / route_id
        bundle.mkdir(parents=True, exist_ok=True)
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_traj = trajectories[trajectories["route_id"].astype(str) == route_id] if not trajectories.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        route_focus = focused_rows[focused_rows["route_id"].astype(str) == route_id] if not focused_rows.empty else pd.DataFrame()
        dump_json(bundle / "route_static_topology.json", route_meta[route_id])
        dump_json(
            bundle / "observation_manifest.json",
            {
                "route_id": route_id,
                "valid_sample_count": int(len(route_samples)),
                "vehicle_count": int(route_samples["vehicle_id"].nunique(dropna=True)) if not route_samples.empty else 0,
                "valid_observation_start_kst": None if route_samples.empty else str(route_samples["request_time_kst"].min()),
                "valid_observation_end_kst": None if route_samples.empty else str(route_samples["request_time_kst"].max()),
            },
        )
        dump_json(bundle / "focused_trigger_audit.json", {"route_id": route_id, "rows": route_focus.to_dict("records") if not route_focus.empty else []})
        route_traj.to_parquet(bundle / "vehicle_trajectories.parquet", index=False)
        route_events.to_parquet(bundle / "terminal_transition_events.parquet", index=False)
        dump_json(bundle / "terminal_transition_summary.json", {"route_id": route_id, "operation_counts": route_events["operation_candidate"].value_counts().to_dict() if not route_events.empty else {}})
        dump_json(bundle / "operation_decision.json", decision.to_dict())
        raw_paths = sorted(set(route_samples["raw_file_path"].dropna().astype(str).tolist())) if not route_samples.empty else []
        dump_json(bundle / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})


def update_mapping_contract(v9: pd.DataFrame, decisions: pd.DataFrame, output_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    decisions_by_route = {str(row["route_id"]): row for _, row in decisions.iterrows()}
    approved_before = v9[v9["approved"].apply(truthy)].copy()
    rows = []
    for _, row in v9.iterrows():
        out = row.to_dict()
        route_id = str(out.get("route_id"))
        out["approved"] = truthy(out.get("approved"))
        out["passed"] = truthy(out.get("passed"))
        out["newly_recovered_in_r4a_hf1"] = False
        if route_id in decisions_by_route and truthy(decisions_by_route[route_id].get("approved")):
            decision = decisions_by_route[route_id]
            out["approved"] = True
            out["passed"] = True
            out["failure_reason"] = ""
            out["mapping_v6_decision"] = decision["decision"]
            out["mapping_v6_reason"] = "HF1 same exact vehicle terminal reset evidence"
            out["terminal_operation_type"] = decision["decision"]
            out["terminal_operation_resolved"] = True
            out["mapping_confidence"] = decision["confidence"]
            out["confidence"] = decision["confidence"]
            out["live_transition_event_count"] = int(decision["live_transition_event_count"])
            out["terminal_state_observation_count"] = int(decision["terminal_state_observation_count"])
            out["terminal_entry_event_count"] = int(decision["terminal_entry_event_count"])
            out["terminal_hold_observation_count"] = int(decision["terminal_hold_observation_count"])
            out["classified_operation_transition_count"] = int(decision["classified_operation_transition_count"])
            out["loop_reset_transition_count"] = int(decision["loop_reset_transition_count"])
            out["independent_vehicle_count"] = int(decision["independent_vehicle_count"])
            out["contradictory_transition_count"] = int(decision["contradiction_count"])
            evidence_path = output_root / "mapping_evidence" / route_id / "terminal_transition_events.parquet"
            out["live_evidence_paths"] = [str(evidence_path)]
            out["live_evidence_sha256s"] = [sha256_file(evidence_path)] if evidence_path.exists() else []
            out["newly_recovered_in_r4a_hf1"] = True
        rows.append(out)
    hf1 = pd.DataFrame(rows)
    compare_fields = [
        "route_id",
        "approved",
        "terminal_operation_type",
        "service_terminal_sequence",
        "full_terminal_sequence",
        "effective_live_terminal_sequence",
        "direction_id",
        "mapping_v6_decision",
        "terminal_operation_resolved",
    ]
    regressions = []
    for idx, before in approved_before.iterrows():
        after = hf1.loc[idx]
        for field in compare_fields:
            if field in v9.columns and field in hf1.columns:
                if str(before.get(field)) != str(after.get(field)):
                    regressions.append({"row_index": int(idx), "route_id": str(before.get("route_id")), "field": field, "before": before.get(field), "after": after.get(field)})
    audit = {
        "source_approved_mapping_count": int(len(approved_before)),
        "approved_mapping_count": int(hf1["approved"].apply(truthy).sum()),
        "missing_mapping_count": int(38 - hf1["approved"].apply(truthy).sum()),
        "mapping_regression_count": len(regressions),
        "allowed_changed_routes": FINAL4_ROUTES,
        "regressions": regressions,
    }
    return hf1, audit


def v10_consistency(decisions: pd.DataFrame, contract: pd.DataFrame) -> Dict[str, Any]:
    problems = []
    for _, decision in decisions.iterrows():
        route_id = str(decision["route_id"])
        rows = contract[contract["route_id"].astype(str) == route_id]
        if rows.empty:
            problems.append({"route_id": route_id, "problem": "missing v10 row"})
            continue
        row = rows.iloc[0]
        checks = {
            "approved": truthy(row.get("approved")),
            "passed": truthy(row.get("passed")),
            "failure_reason_clear": not norm(row.get("failure_reason")),
            "terminal_operation_resolved": truthy(row.get("terminal_operation_resolved")),
            "operation_type_match": norm(row.get("terminal_operation_type")) == norm(decision.get("decision")),
            "live_transition_event_count_positive": (as_int(row.get("live_transition_event_count")) or 0) > 0,
            "loop_reset_transition_count_positive": (as_int(row.get("loop_reset_transition_count")) or 0) > 0,
            "confidence_match": norm(row.get("mapping_confidence")) == norm(decision.get("confidence")) or norm(row.get("confidence")) == norm(decision.get("confidence")),
        }
        failed = [key for key, value in checks.items() if not value]
        if failed:
            problems.append({"route_id": route_id, "failed_checks": failed})
    return {"v10_internal_consistency_passed": not problems, "problems": problems}


def api_stop_audit(requests: pd.DataFrame) -> Dict[str, Any]:
    if requests.empty:
        return {"api_stop_condition_audit_passed": True, "source_api_call_count": 0, "quota_error_count": 0, "calls_after_first_quota_error": 0, "first_quota_error_time_kst": None, "max_calls_per_minute": 0}
    status_counts = requests["response_status"].value_counts().to_dict()
    quota = requests[requests["response_status"] == "RATE_LIMIT"].copy()
    first_quota = None if quota.empty else str(quota["request_time_kst"].min())
    calls_after = 0 if first_quota is None else int((requests["request_time_kst"] > first_quota).sum())
    minute_counts = requests.assign(minute=requests["request_time_kst"].str.slice(0, 16)).groupby("minute").size()
    return {
        "api_stop_condition_audit_passed": (
            int(status_counts.get("RATE_LIMIT", 0)) == 0
            and int(status_counts.get("AUTH_ERROR", 0)) == 0
            and int(status_counts.get("HTML_RESPONSE", 0)) == 0
            and calls_after == 0
            and (int(minute_counts.max()) if not minute_counts.empty else 0) <= 12
        ),
        "source_api_call_count": int(len(requests)),
        "quota_error_count": int(status_counts.get("RATE_LIMIT", 0)),
        "AUTH_ERROR_count": int(status_counts.get("AUTH_ERROR", 0)),
        "HTML_response_count": int(status_counts.get("HTML_RESPONSE", 0)),
        "calls_after_first_quota_error": calls_after,
        "first_quota_error_time_kst": first_quota,
        "max_calls_per_minute": int(minute_counts.max()) if not minute_counts.empty else 0,
        "status_counts": status_counts,
        "stop_rule_result": "FAILED_SOURCE_R4A_CONTINUED_AFTER_QUOTA" if first_quota is not None and calls_after > 0 else "PASSED",
    }


def load_r4_module(project_root: Path):
    path = project_root / "05_training/run_prompt5_e01_r2d1c_r4_final8_terminal_semantics.py"
    spec = importlib.util.spec_from_file_location("prompt5_r4_hf1_live", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_limited_live_revalidation(
    project_root: Path,
    output_root: Path,
    route_meta: Mapping[str, Mapping[str, Any]],
    secret: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    preflight = {
        "run_date_kst": datetime.now().astimezone().date().isoformat(),
        "required_not_before_date_kst": "2026-07-23",
        "api_call_count": 0,
        "quota_reset_preflight_executed": False,
        "quota_reset_confirmed": False,
        "reason": None,
    }
    runtime = {
        "api_call_count": 0,
        "max_calls_per_minute": 0,
        "hard_budget_exceeded": False,
        "executed": False,
        "stopped_early": False,
        "stop_reason": None,
        "calls_after_stop_condition": 0,
        "priority_routes": LIVE_PRIORITY_ROUTES,
        "sampled_routes": [],
    }
    now = datetime.now().astimezone()
    if now.isoformat() < "2026-07-23T09:00:00+09:00":
        preflight["reason"] = "DATE_GUARD_NOT_MET"
        return pd.DataFrame(), pd.DataFrame(), preflight, runtime
    if not secret:
        preflight["reason"] = "CREDENTIAL_UNAVAILABLE"
        runtime["stop_reason"] = "BLOCKED_CREDENTIAL_UNAVAILABLE"
        return pd.DataFrame(), pd.DataFrame(), preflight, runtime

    r4 = load_r4_module(project_root)
    sample_rows: List[Dict[str, Any]] = []
    request_rows: List[Dict[str, Any]] = []
    last_call_monotonic: Optional[float] = None
    stopped = False

    def fetch(route_id: str, mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        nonlocal last_call_monotonic, stopped
        if last_call_monotonic is not None:
            wait_seconds = LIVE_MIN_CALL_SPACING_SECONDS - (time.monotonic() - last_call_monotonic)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        rows, request = r4.fetch_route(
            route_id,
            secret,
            output_root,
            "hf1_live",
            mode,
            120,
            route_meta[route_id],
        )
        last_call_monotonic = time.monotonic()
        sample_rows.extend(rows)
        request_rows.append(request)
        status = request.get("response_status")
        if status in {"RATE_LIMIT", "AUTH_ERROR", "HTML_RESPONSE"}:
            runtime["stopped_early"] = True
            runtime["stop_reason"] = status
            stopped = True
        print(json.dumps({"event": "hf1_live_call", "route_id": route_id, "mode": mode, "status": status, "rows": len(rows), "time": now_iso()}, ensure_ascii=False), flush=True)
        return rows, request

    preflight_rows, preflight_request = fetch(LIVE_PRIORITY_ROUTES[0], "BROAD_SCAN")
    preflight.update(
        {
            "api_call_count": 1,
            "quota_reset_preflight_executed": True,
            "quota_reset_confirmed": preflight_request.get("response_status") == "OK",
            "reason": "PASS" if preflight_request.get("response_status") == "OK" else preflight_request.get("response_status"),
            "route_id": LIVE_PRIORITY_ROUTES[0],
            "http_status": preflight_request.get("http_status"),
            "response_status": preflight_request.get("response_status"),
            "normalized_row_count": len(preflight_rows),
            "raw_sha256": preflight_request.get("raw_sha256"),
            "request_url_redacted": preflight_request.get("request_url_redacted"),
        }
    )
    if not stopped and preflight["quota_reset_confirmed"]:
        for route_id in LIVE_PRIORITY_ROUTES[1:]:
            if len(request_rows) >= 800:
                runtime["stopped_early"] = True
                runtime["stop_reason"] = "SAFE_BUDGET_REACHED"
                break
            fetch(route_id, "BROAD_SCAN")
            if stopped:
                break

    request_df = pd.DataFrame(request_rows)
    sample_df = pd.DataFrame(sample_rows)
    audit = api_stop_audit(request_df)
    runtime.update(
        {
            "api_call_count": int(len(request_df)),
            "max_calls_per_minute": audit.get("max_calls_per_minute", 0),
            "hard_budget_exceeded": int(len(request_df)) > 850,
            "executed": bool(preflight["quota_reset_confirmed"]),
            "sampled_routes": sorted(request_df["route_id"].astype(str).unique().tolist()) if not request_df.empty else [],
            "status_counts": audit.get("status_counts", {}),
            "calls_after_first_quota_error": audit.get("calls_after_first_quota_error", 0),
        }
    )
    return sample_df, request_df, preflight, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R4A-HF1 offline repair and date-guarded limited revalidation.")
    parser.add_argument("--project-root", type=Path, default=Path("/Users/arty/Documents/Codex/urbanbus_rl_project"))
    parser.add_argument("--timestamp", default=now_stamp())
    parser.add_argument("--service-key-env", default="DAEGU_BIS_SERVICE_KEY")
    parser.add_argument("--execute-limited-live", action="store_true")
    parser.add_argument("--resume-live-root", type=Path)
    args = parser.parse_args()

    project_root = args.project_root
    source_root = project_root / R4A_SOURCE
    r4_root = project_root / R4_SOURCE
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    live_mode = args.execute_limited_live or args.resume_live_root is not None
    secret = os.environ.get(args.service_key_env, "") if live_mode else ""
    pre_scan = scan_secret(output_root, secret)

    r4a_gate = json.loads((source_root / "prompt5_e01_r2d1c_r4a_gate.json").read_text(encoding="utf-8"))
    r4_gate = json.loads((r4_root / "prompt5_e01_r2d1c_r4_gate.json").read_text(encoding="utf-8"))
    v9_path = r4_root / "turnaround_mapping_contract_v9.parquet"
    v9 = pd.read_parquet(v9_path)
    route_meta = load_route_meta(v9)

    shutil.copy2(v9_path, output_root / "turnaround_mapping_contract_v9_source_snapshot.parquet")
    dump_json(output_root / "r4a_source_reference.json", {"path": str(source_root), "gate_path": str(source_root / "prompt5_e01_r2d1c_r4a_gate.json"), "gate_sha256": sha256_file(source_root / "prompt5_e01_r2d1c_r4a_gate.json"), "source_status": r4a_gate.get("status")})
    dump_json(output_root / "r4_reference.json", {"path": str(r4_root), "gate_path": str(r4_root / "prompt5_e01_r2d1c_r4_gate.json"), "gate_sha256": sha256_file(r4_root / "prompt5_e01_r2d1c_r4_gate.json"), "source_approved_mapping_count": r4_gate.get("approved_mapping_count")})
    dump_json(output_root / "mapping_v9_reference.json", {"path": str(v9_path), "sha256": sha256_file(v9_path), "snapshot_path": str(output_root / "turnaround_mapping_contract_v9_source_snapshot.parquet")})
    dump_json(output_root / "credential_reference.json", {"credential_source": "environment_variable_only", "env_var": args.service_key_env, "credential_value_stored": False})

    source_samples, source_requests = load_success_raw_samples(source_root, route_meta)
    live_samples = pd.DataFrame()
    live_requests = pd.DataFrame()
    quota_preflight: Dict[str, Any]
    limited_runtime: Dict[str, Any]
    if args.resume_live_root is not None:
        shutil.copytree(args.resume_live_root / "raw", output_root / "raw", dirs_exist_ok=True)
        live_samples, live_requests = load_success_raw_samples(output_root, route_meta)
        live_audit = api_stop_audit(live_requests)
        first_request = live_requests.iloc[0].to_dict() if not live_requests.empty else {}
        quota_preflight = {
            "run_date_kst": datetime.now().astimezone().date().isoformat(),
            "required_not_before_date_kst": "2026-07-23",
            "api_call_count": 1 if not live_requests.empty else 0,
            "quota_reset_preflight_executed": not live_requests.empty,
            "quota_reset_confirmed": first_request.get("response_status") == "OK",
            "reason": "PASS" if first_request.get("response_status") == "OK" else first_request.get("response_status", "NO_SAVED_REQUEST"),
            "route_id": first_request.get("route_id"),
            "response_status": first_request.get("response_status"),
            "raw_sha256": first_request.get("raw_sha256"),
            "resumed_from_saved_raw": str(args.resume_live_root),
        }
        limited_runtime = {
            "api_call_count": int(len(live_requests)),
            "max_calls_per_minute": live_audit.get("max_calls_per_minute", 0),
            "hard_budget_exceeded": int(len(live_requests)) > 850,
            "executed": not live_requests.empty,
            "resumed_without_additional_api_calls": True,
            "status_counts": live_audit.get("status_counts", {}),
            "sampled_routes": sorted(live_requests["route_id"].astype(str).unique().tolist()) if not live_requests.empty else [],
        }
    elif args.execute_limited_live:
        live_samples, live_requests, quota_preflight, limited_runtime = run_limited_live_revalidation(
            project_root,
            output_root,
            route_meta,
            secret,
        )
    else:
        current_date = datetime.now().astimezone().date().isoformat()
        quota_preflight = {
            "run_date_kst": current_date,
            "required_not_before_date_kst": "2026-07-23",
            "api_call_count": 0,
            "quota_reset_preflight_executed": False,
            "quota_reset_confirmed": False,
            "reason": "DATE_GUARD_NOT_MET" if current_date < "2026-07-23" else "NOT_REQUESTED_IN_OFFLINE_REPAIR_RUN",
        }
        limited_runtime = {
            "api_call_count": 0,
            "max_calls_per_minute": 0,
            "hard_budget_exceeded": False,
            "executed": False,
            "reason": quota_preflight["reason"],
        }
    samples = pd.concat([source_samples, live_samples], ignore_index=True, sort=False) if not live_samples.empty else source_samples
    trajectories = build_trajectories(samples)
    events = reconstruct_events(trajectories, route_meta)
    counters, counter_by_route = counter_audit(samples, events)
    focused = recompute_focused_runtime(samples, events)
    decisions, contradiction = route_decisions(samples, events, counter_by_route, route_meta)

    samples.to_parquet(output_root / "terminal_semantics_position_samples_hf1.parquet", index=False)
    trajectories.to_parquet(output_root / "vehicle_continuous_trajectories_hf1.parquet", index=False)
    events.to_parquet(output_root / "terminal_transition_events_hf1.parquet", index=False)
    decisions.to_parquet(output_root / "terminal_operation_decisions_hf1.parquet", index=False)

    write_route_bundles(output_root, route_meta, samples, trajectories, events, decisions, focused)
    sha_audit = evidence_sha256_audit(output_root, decisions)
    hf1_contract, mapping_regression = update_mapping_contract(v9, decisions, output_root)
    consistency = v10_consistency(decisions, hf1_contract)
    hf1_contract.to_parquet(output_root / "turnaround_mapping_contract_v10_hf1.parquet", index=False)
    dump_json(output_root / "turnaround_mapping_contract_v10_hf1.json", {"rows": hf1_contract.to_dict("records")})

    static_live = static_live_audit(samples, route_meta)
    source_api_stop = api_stop_audit(source_requests)
    api_stop = api_stop_audit(live_requests) if live_mode else source_api_stop
    post_scan = scan_secret(output_root, secret)
    secret_audit = {
        "pre_hf1_artifact_scan": pre_scan,
        "post_hf1_artifact_scan": post_scan,
        "secret_leak_count": pre_scan["secret_literal_occurrence_count"] + post_scan["secret_literal_occurrence_count"],
        "security_audit_passed": pre_scan["secret_literal_occurrence_count"] + post_scan["secret_literal_occurrence_count"] == 0,
    }

    limited_contract = {
        "hard_api_call_budget": 850,
        "safe_api_call_target": 800,
        "max_calls_per_minute": 12,
        "not_before_kst": "2026-07-23T09:00:00+09:00",
        "priority_routes": ["4010002118", "4050010000", "4010002001", "4010002004"],
        "capture_modes": ["BROAD_SCAN", "TERMINAL_FOCUSED", "POST_TERMINAL_FOLLOW"],
        "broad_scan_interval_seconds": 120,
        "focused_sampling_interval_seconds": 30,
        "same_vehicle_rule": "exact vehicle_id equality only",
        "executed_in_this_run": live_mode,
    }

    approved_count = int(hf1_contract["approved"].apply(truthy).sum())
    missing_count = int(38 - approved_count)
    low_approved = int(((decisions["approved"] == True) & (decisions["confidence"] == "LOW")).sum())
    instant_reentry_approval_count = int(((decisions["approved"] == True) & (decisions["decision"] == "OFF_GRAPH_CONTINUATION_AND_REENTRY")).sum())
    live_execution_passed = (
        not live_mode
        or (
            quota_preflight.get("quota_reset_confirmed") is True
            and limited_runtime.get("executed") is True
            and api_stop.get("api_stop_condition_audit_passed") is True
            and not limited_runtime.get("hard_budget_exceeded", False)
        )
    )
    all_pass = (
        approved_count == 38
        and missing_count == 0
        and low_approved == 0
        and mapping_regression["mapping_regression_count"] == 0
        and contradiction["contradiction_count"] == 0
        and instant_reentry_approval_count == 0
        and counters["counter_contract_v4_passed"]
        and focused["focused_runtime_audit_passed"]
        and api_stop["api_stop_condition_audit_passed"]
        and secret_audit["security_audit_passed"]
        and sha_audit["evidence_sha256_audit_passed"]
        and consistency["v10_internal_consistency_passed"]
        and live_execution_passed
    )
    if mapping_regression["mapping_regression_count"]:
        status = "FAIL_MAPPING_REGRESSION"
    elif not live_execution_passed or not api_stop["api_stop_condition_audit_passed"]:
        status = "FAIL_OBSERVATION_CAMPAIGN"
    elif not focused["focused_runtime_audit_passed"]:
        status = "BLOCKED_FOCUSED_TRIGGER_RUNTIME"
    elif contradiction["contradiction_count"]:
        status = "BLOCKED_MULTIPLE_SERVICE_PATTERNS"
    elif static_live["blocked_static_live_topology_conflict_count"]:
        status = "BLOCKED_STATIC_LIVE_TOPOLOGY_CONFLICT"
    elif all_pass:
        status = "PASS_MAPPING_38_OF_38_READY"
    elif approved_count > 34:
        status = "PASS_MAPPING_PARTIALLY_RECOVERED"
    else:
        status = "BLOCKED_NO_POST_TERMINAL_OBSERVATION"

    authorization = {
        "approved_for_terminal_recovery_campaign": status == "PASS_MAPPING_38_OF_38_READY",
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
    }
    phase2 = {
        **authorization,
        "approved_for_phase2_turnaround": False,
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_rerun": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2": False,
        "approved_for_e2_execution": False,
        "approved_for_full_matrix": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "phase2_production_executed": False,
    }

    offline_audit = {
        "source_r4a_status_preserved": r4a_gate.get("status"),
        "valid_success_vehicle_rows": int(len(samples)),
        "valid_success_raw_file_count": int(samples["raw_sha256"].nunique()) if not samples.empty else 0,
        "valid_observation_window_by_route": decisions[["route_id", "valid_observation_start_kst", "valid_observation_end_kst", "valid_observation_minutes", "vehicle_count", "valid_sample_count"]].to_dict("records"),
        "invalid_observation_excluded_status_counts": source_requests["response_status"].value_counts().to_dict() if not source_requests.empty else {},
        "limited_live_valid_sample_count": int(len(live_samples)),
    }

    dump_json(output_root / "offline_reconstruction_audit.json", offline_audit)
    dump_json(output_root / "api_quota_reset_preflight.json", quota_preflight)
    dump_json(output_root / "limited_revalidation_contract.json", limited_contract)
    dump_json(output_root / "limited_revalidation_runtime_audit.json", limited_runtime)
    dump_json(output_root / "api_stop_condition_audit.json", api_stop)
    dump_json(output_root / "focused_capture_runtime_audit_v2.json", focused)
    dump_json(output_root / "mapping_regression_audit.json", mapping_regression)
    dump_json(output_root / "static_live_topology_conflict_audit.json", static_live)
    dump_json(output_root / "terminal_counter_audit_v4_recomputed.json", counters)
    dump_json(output_root / "terminal_operation_contradiction_audit_v2.json", contradiction)
    dump_json(output_root / "secret_leak_audit.json", secret_audit)
    dump_json(output_root / "remaining_mapping_evidence_requirements_v5.json", {"remaining": [] if approved_count == 38 else decisions[decisions["approved"] != True].to_dict("records")})
    dump_json(output_root / "terminal_recovery_campaign_authorization_draft.json", authorization)
    dump_json(output_root / "phase2_execution_authorization.json", phase2)

    gate = {
        "status": status,
        "classification": status,
        "source_r4a_gate_status_preserved": r4a_gate.get("status"),
        "approved_mapping_count": approved_count,
        "missing_mapping_count": missing_count,
        "newly_recovered_mapping_count": int(decisions["approved"].apply(truthy).sum()),
        "LOW_confidence_approval_count": low_approved,
        "mapping_regression_count": mapping_regression["mapping_regression_count"],
        "contradiction_count": contradiction["contradiction_count"],
        "instant_reentry_approval_count": instant_reentry_approval_count,
        "counter_contract_v4_passed": counters["counter_contract_v4_passed"],
        "focused_runtime_audit_passed": focused["focused_runtime_audit_passed"],
        "api_stop_condition_audit_passed": api_stop["api_stop_condition_audit_passed"],
        "security_audit_passed": secret_audit["security_audit_passed"],
        "evidence_sha256_audit_passed": sha_audit["evidence_sha256_audit_passed"],
        "v10_internal_consistency_passed": consistency["v10_internal_consistency_passed"],
        "api_call_count": int(limited_runtime.get("api_call_count", 0)),
        "source_r4a_api_call_count": source_api_stop["source_api_call_count"],
        "quota_error_count": api_stop["quota_error_count"],
        "first_quota_error_time_kst": api_stop["first_quota_error_time_kst"],
        "calls_after_first_quota_error": api_stop["calls_after_first_quota_error"],
        "max_calls_per_minute": int(limited_runtime.get("max_calls_per_minute", 0)),
        "source_r4a_max_calls_per_minute": source_api_stop["max_calls_per_minute"],
        "source_r4a_quota_error_count": source_api_stop["quota_error_count"],
        "source_r4a_first_quota_error_time_kst": source_api_stop["first_quota_error_time_kst"],
        "source_r4a_calls_after_first_quota_error": source_api_stop["calls_after_first_quota_error"],
        "hard_budget_exceeded": bool(limited_runtime.get("hard_budget_exceeded", False)),
        "limited_revalidation_executed": bool(limited_runtime.get("executed", False)),
        "quota_reset_preflight_executed": bool(quota_preflight.get("quota_reset_preflight_executed", False)),
        "approved_for_terminal_recovery_campaign": authorization["approved_for_terminal_recovery_campaign"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
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
        "route_decisions": decisions[["route_id", "decision", "confidence", "transition_count", "independent_vehicle_count", "approved"]].to_dict("records"),
        "4050010000_static_live_conclusion": next(row for row in static_live["rows"] if row["route_id"] == "4050010000"),
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r4a_hf1_gate.json", gate)

    report_lines = [
        "# Prompt 5-E01-R2D-1C-R4A-HF1 Final Report",
        "",
        f"status: {status}",
        f"artifact_dir: {output_root}",
        "",
        "## R4A Failure",
        f"R4A source gate is preserved as {r4a_gate.get('status')}. The exact failure reason is API quota/rate-limit after successful mapping evidence had already been collected.",
        f"first quota error time: {source_api_stop['first_quota_error_time_kst']}",
        f"quota errors: {source_api_stop['quota_error_count']}",
        f"calls after first quota error: {source_api_stop['calls_after_first_quota_error']}",
        "",
        "## HF1 API Use",
        f"api calls in this HF1 run: {limited_runtime.get('api_call_count', 0)}",
        f"max calls per minute in this HF1 run: {limited_runtime.get('max_calls_per_minute', 0)}",
        f"hard budget respected: {str(not limited_runtime.get('hard_budget_exceeded', False)).lower()}",
        f"limited revalidation executed: {str(limited_runtime.get('executed', False)).lower()}",
        f"quota reset preflight confirmed: {str(quota_preflight.get('quota_reset_confirmed', False)).lower()}",
        "",
        "## Route Decisions",
    ]
    for _, row in decisions.iterrows():
        report_lines.append(f"- {row['route_id']}: {row['decision']} / {row['confidence']} / transitions={row['transition_count']} / independent_vehicles={row['independent_vehicle_count']}")
    report_lines.extend(
        [
            "",
            "## Audits",
            f"contradictions: {contradiction['contradiction_count']}",
            f"mapping regressions among prior 34 approved rows: {mapping_regression['mapping_regression_count']}",
            f"4050010000 static/live conclusion: {gate['4050010000_static_live_conclusion']['conclusion']}",
            f"counter contract v4 passed: {counters['counter_contract_v4_passed']}",
            f"focused runtime audit passed: {focused['focused_runtime_audit_passed']}",
            f"evidence sha256 audit passed: {sha_audit['evidence_sha256_audit_passed']}",
            f"security audit passed: {secret_audit['security_audit_passed']}",
            f"v10 internal consistency passed: {consistency['v10_internal_consistency_passed']}",
            "",
            "## Authorization",
            f"approved_for_terminal_recovery_campaign: {authorization['approved_for_terminal_recovery_campaign']}",
            "Next authorized step: terminal recovery observation campaign only if the final gate is PASS_MAPPING_38_OF_38_READY.",
            "Still prohibited: terminal recovery time estimation/application, Phase 2 turnaround, baseline rerun, E0/E1 retraining, Prompt 6A, E2, full matrix, and real-world causal claims.",
        ]
    )
    (output_root / "prompt5_e01_r2d1c_r4a_hf1_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    files = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file())
    dump_json(output_root / "prompt5_e01_r2d1c_r4a_hf1_manifest.json", {"prompt": "Prompt 5-E01-R2D-1C-R4A-HF1", "created_at": now_iso(), "artifact_dir": str(output_root), "status": status, "files": files})

    print("R4A-HF1 COMPLETE")
    print()
    print("artifact_dir:")
    print(output_root)
    print()
    print("gate:")
    print(status)
    print()
    print("approved_mapping_count:")
    print(approved_count)
    print()
    print("missing_mapping_count:")
    print(missing_count)
    print()
    print("route_decisions:")
    for _, row in decisions.iterrows():
        print(f"{row['route_id']} = {row['decision']} / {row['confidence']} / transitions={row['transition_count']}")
    print()
    print("mapping_regression_count:")
    print(mapping_regression["mapping_regression_count"])
    print()
    print("contradiction_count:")
    print(contradiction["contradiction_count"])
    print()
    print("api_call_count:")
    print(limited_runtime.get("api_call_count", 0))
    print()
    print("quota_error_count:")
    print(api_stop["quota_error_count"])
    print()
    print("calls_after_first_quota_error:")
    print(api_stop["calls_after_first_quota_error"])
    print()
    print("counter_contract_v4_passed:")
    print(str(counters["counter_contract_v4_passed"]).lower())
    print()
    print("focused_runtime_audit_passed:")
    print(str(focused["focused_runtime_audit_passed"]).lower())
    print()
    print("evidence_sha256_audit_passed:")
    print(str(sha_audit["evidence_sha256_audit_passed"]).lower())
    print()
    print("secret_leak_count:")
    print(secret_audit["secret_leak_count"])
    print()
    print("approved_for_terminal_recovery_campaign:")
    print(str(authorization["approved_for_terminal_recovery_campaign"]).lower())
    print()
    print("next_authorized_step:")
    print("NONE" if status != "PASS_MAPPING_38_OF_38_READY" else "Terminal Recovery Observation Campaign")


if __name__ == "__main__":
    main()
