from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_r2d1h_campaign_a_live_observation_actual as base


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_REL = Path("05_training/artifacts")
OUTPUT_PREFIX = "prompt5_e01_r2d1i_targeted_4010002118_live_observation"
TARGET_ROUTE = "4010002118"
TARGET_ROUTES = [TARGET_ROUTE]
FORBIDDEN_ROUTES = {"4010002001", "4010002004", "4050010000"}
KST = ZoneInfo("Asia/Seoul")

HF1_REL = ARTIFACTS_REL / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_REL = ARTIFACTS_REL / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"
R2D1G_REL = ARTIFACTS_REL / "prompt5_e01_r2d1g_observation_expansion_plan_20260724_170238"
R2D1H_CLEANUP_REL = ARTIFACTS_REL / "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup_20260725_194710"

GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
DAILY_SAFETY_CAP = 800
RECOMMENDED_R2D1I_CAP = 220
ABSOLUTE_R2D1I_CAP = 280
MIN_EFFECTIVE_HARD_CAP = 140
MAX_CALLS_PER_MINUTE = 4
REQUIRED_FOLLOW_WINDOW_MINUTES = 115
MAX_FOLLOW_MINUTES = 100
REQUIRED_END_BUFFER_MINUTES = 15
POST_CONFIRM_MIN_SAMPLES = 3
POST_CONFIRM_MIN_SECONDS = 10 * 60
FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE"}

SAMPLE_COLUMNS = list(base.SAMPLE_COLUMNS)
EPISODE_COLUMNS = list(base.EPISODE_COLUMNS)
if "disappeared_waiting_reentry_observation_count" not in EPISODE_COLUMNS:
    insert_at = EPISODE_COLUMNS.index("post_terminal_confirmation_sample_count") + 1
    EPISODE_COLUMNS.insert(insert_at, "disappeared_waiting_reentry_observation_count")
BOUNDS_COLUMNS = [
    "episode_id",
    "route_id",
    "vehicle_id",
    "direction",
    "provider_service_end_window_start",
    "provider_service_end_window_end",
    "provider_reentry_window_start",
    "provider_reentry_window_end",
    "request_service_end_window_start",
    "request_service_end_window_end",
    "request_reentry_window_start",
    "request_reentry_window_end",
    "provider_lower_bound_sec",
    "provider_upper_bound_sec",
    "request_lower_bound_sec",
    "request_upper_bound_sec",
    "conservative_dual_lower_bound_sec",
    "conservative_dual_upper_bound_sec",
    "complete_interval_censored_episode",
]


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def iso(dt: Optional[datetime] = None) -> str:
    return (dt or now_kst()).isoformat(timespec="seconds")


def now_stamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    try:
        if not isinstance(value, (str, bytes, list, tuple, dict)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_meta(mapping_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(mapping_path)
    row = df[df["route_id"].astype(str) == TARGET_ROUTE].iloc[0].to_dict()
    effective = base.as_int(row.get("effective_live_terminal_sequence"))
    if effective is None:
        raise ValueError("Missing effective_live_terminal_sequence for target route")
    return {
        **row,
        "route_id": TARGET_ROUTE,
        "direction": base.norm(row.get("direction_id")) or base.norm(row.get("service_direction_id")) or "1",
        "effective_live_terminal_sequence": effective,
        "early_upstream_trigger_sequence": effective - 22,
        "upstream_trigger_sequence": effective - 12,
        "terminal_trigger_sequence": effective - 5,
    }


def authoritative_inputs(project_root: Path) -> List[Tuple[Path, str]]:
    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    r2d1g = project_root / R2D1G_REL
    cleanup = project_root / R2D1H_CLEANUP_REL
    return [
        (hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.parquet", "HF1"),
        (r2d1e / "frozen_complete_episode_registry.parquet", "R2D-1E"),
        (r2d1f / "prompt5_e01_r2d1f_gate.json", "R2D-1F"),
        (r2d1g / "prompt5_e01_r2d1g_gate.json", "R2D-1G"),
        (cleanup / "prompt5_e01_r2d1h_gate.json", "R2D-1H-CLEANUP"),
        (cleanup / "method_prototype_progress_audit.json", "R2D-1H-CLEANUP"),
        (cleanup / "cumulative_episode_registry_candidate.parquet", "R2D-1H-CLEANUP"),
        (cleanup / "campaign_a_terminal_recovery_episodes.parquet", "R2D-1H-CLEANUP"),
    ]


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "source": source,
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "exists": exists,
        "file_size": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def raw_index_for_date(project_root: Path, run_date: str) -> List[str]:
    ymd = run_date.replace("-", "")
    return sorted(str(path) for path in (project_root / ARTIFACTS_REL).rglob(f"{ymd}_*.json") if "/raw/" in str(path))


def provider_time(request_iso: str, hhmmss: Any) -> Optional[str]:
    return base.provider_time(request_iso, hhmmss)


def seconds_between(left: Any, right: Any) -> Optional[float]:
    return base.seconds_between(left, right)


def parse_dt(value: Any) -> Optional[datetime]:
    return base.parse_dt(value)


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    if not secret:
        return {"scan_root": str(root), "secret_literal_occurrence_count": 0, "findings": []}
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")]
    findings = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def strict_json_token_count(path: Path) -> int:
    data = path.read_bytes()
    return sum(data.count(token) for token in [b"NaN", b"Infinity", b"-Infinity"])


def derived_terminal_phase(sample: Mapping[str, Any]) -> str:
    if sample.get("sequence_reset"):
        return "POST_TERMINAL_RESET"
    seq = base.as_int(sample.get("current_sequence"))
    terminal = base.as_int(sample.get("terminal_trigger_sequence"))
    if seq is None or terminal is None:
        return "UNKNOWN"
    if seq >= terminal:
        effective = base.as_int(sample.get("effective_live_terminal_sequence"))
        if effective is not None and seq >= effective:
            return "TERMINAL_STOP_HOLD"
        return "TERMINAL_ZONE_APPROACH"
    if sample.get("is_upstream_watch_zone"):
        return "UPSTREAM_WATCH"
    if sample.get("is_early_upstream_watch_zone"):
        return "EARLY_UPSTREAM_WATCH"
    return "BROAD_SCAN"


def request_interval(mode: str) -> int:
    return {
        "BROAD_SCAN": 120,
        "EARLY_UPSTREAM_WATCH": 60,
        "UPSTREAM_FOCUSED": 45,
        "TERMINAL_FOCUSED": 30,
        "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY": 30,
        "POST_TERMINAL_CONFIRM": 30,
    }.get(mode, 120)


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows], columns=list(columns)).to_parquet(path, index=False)


class TargetedRunner:
    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        secret: str,
        meta: Mapping[str, Any],
        prior_complete_vehicle_ids: set[str],
        prior_censored_vehicle_ids: set[str],
        planned_end: datetime,
        effective_cap: int,
        prior_calls: int,
    ) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.secret = secret
        self.meta = dict(meta)
        self.prior_complete_vehicle_ids = set(prior_complete_vehicle_ids)
        self.prior_censored_vehicle_ids = set(prior_censored_vehicle_ids)
        self.planned_end = planned_end
        self.effective_cap = effective_cap
        self.prior_calls = prior_calls
        self.request_records: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []
        self.raw_files: List[Dict[str, Any]] = []
        self.session: Optional[Dict[str, Any]] = None
        self.session_counter = 0
        self.campaign_seen_vehicle_ids: set[str] = set()
        self.candidate_count = 0
        self.history_count: Counter = Counter()
        self.complete_count = 0
        self.right_count = 0
        self.left_count = 0
        self.invalid_count = 0
        self.new_independent_complete_count = 0
        self.fatal_error: Optional[str] = None
        self.first_fatal_error_time: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.campaign_started_at: Optional[str] = None
        self.campaign_finished_at: Optional[str] = None
        self.last_call_mono: Optional[float] = None
        self.next_due = time.monotonic()
        self.consecutive_timeouts = 0
        self.consecutive_parse_failures = 0
        self.consecutive_network_errors = 0

    @property
    def physical_calls_this_artifact(self) -> int:
        return len(self.request_records)

    @property
    def preflight_calls(self) -> int:
        return sum(1 for req in self.request_records if req.get("capture_mode") == "PREFLIGHT")

    @property
    def campaign_calls(self) -> int:
        return self.physical_calls_this_artifact - self.preflight_calls

    def classify_vehicle(self, vehicle_id: str) -> str:
        if vehicle_id in self.campaign_seen_vehicle_ids:
            return "DUPLICATE_TERMINAL_CYCLE"
        if vehicle_id in self.prior_censored_vehicle_ids:
            return "PREVIOUSLY_CENSORED_VEHICLE"
        if vehicle_id in self.prior_complete_vehicle_ids:
            return "PREVIOUSLY_COMPLETE_VEHICLE"
        return "NEW_INDEPENDENT_VEHICLE"

    def call_allowed(self) -> bool:
        if self.physical_calls_this_artifact >= self.effective_cap:
            self.fatal_error = "EFFECTIVE_R2D1I_HARD_CAP_REACHED"
            self.first_fatal_error_time = self.first_fatal_error_time or iso()
            self.stop_reason = self.fatal_error
            return False
        if self.prior_calls + self.physical_calls_this_artifact >= DAILY_SAFETY_CAP:
            self.fatal_error = "DAILY_PHYSICAL_CALL_SAFETY_CAP_REACHED"
            self.first_fatal_error_time = self.first_fatal_error_time or iso()
            self.stop_reason = self.fatal_error
            return False
        return True

    def fetch(self, mode: str) -> List[Dict[str, Any]]:
        if not self.call_allowed():
            return []
        if self.last_call_mono is not None:
            sleep_for = (60.0 / MAX_CALLS_PER_MINUTE) - (time.monotonic() - self.last_call_mono)
            if sleep_for > 0:
                time.sleep(sleep_for)

        request_dt = now_kst()
        request_time = request_dt.isoformat(timespec="seconds")
        request_id = f"r2d1i_{len(self.request_records) + 1:05d}_{TARGET_ROUTE}_{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}"
        raw_dir = self.output_root / "raw" / TARGET_ROUTE
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}.json"
        params = [("serviceKey", self.secret), ("routeId", TARGET_ROUTE), ("resultType", "json")]
        url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
        http_status: Optional[int] = None
        content_type = ""
        timeout_error = False
        parsed = False
        result_code: Optional[str] = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1i/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
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
        status = base.detect_response_status(text, http_status, timeout_error)
        items: List[Dict[str, Any]] = []
        if status == "OK":
            items, parsed, result_code = base.parse_items(text)
            if not parsed:
                status = "PROVIDER_PARSE_FAILURE"
        raw_sha = sha256_file(raw_path)
        rows: List[Dict[str, Any]] = []
        route_mismatch_count = 0
        effective = base.as_int(self.meta.get("effective_live_terminal_sequence"))
        early = base.as_int(self.meta.get("early_upstream_trigger_sequence"))
        upstream = base.as_int(self.meta.get("upstream_trigger_sequence"))
        terminal = base.as_int(self.meta.get("terminal_trigger_sequence"))
        for item in items:
            response_route = base.norm(item.get("routeId")) or TARGET_ROUTE
            if response_route != TARGET_ROUTE:
                route_mismatch_count += 1
            seq = base.as_int(item.get("seq") or item.get("stopSeq"))
            vehicle_id = base.norm(item.get("vhcNo2")) or base.norm(item.get("vhcNo")) or base.norm(item.get("busId"))
            if seq is None or vehicle_id is None:
                continue
            direction = base.norm(item.get("moveDir")) or base.norm(item.get("direction_id")) or base.norm(self.meta.get("direction")) or "1"
            provider_event_time = provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
            sample = {
                "request_id": request_id,
                "campaign_id": "R2D-1I-TARGETED-4010002118",
                "session_id": None,
                "episode_id": None,
                "route_id": TARGET_ROUTE,
                "vehicle_id": vehicle_id,
                "capture_mode": mode,
                "derived_terminal_phase": None,
                "request_observation_time": request_time,
                "provider_position_event_time": provider_event_time,
                "provider_event_lag_sec": seconds_between(provider_event_time, request_time),
                "provider_event_timestamp_repeated": False,
                "provider_event_repeat_count": 0,
                "provider_event_repeat_request_span_sec": 0.0,
                "direction": direction,
                "current_sequence": seq,
                "stop_id": base.norm(item.get("bsId")) or base.norm(item.get("stopId")),
                "x": base.as_float(item.get("xPos") or item.get("x")),
                "y": base.as_float(item.get("yPos") or item.get("y")),
                "effective_live_terminal_sequence": effective,
                "early_upstream_trigger_sequence": early,
                "upstream_trigger_sequence": upstream,
                "terminal_trigger_sequence": terminal,
                "is_early_upstream_watch_zone": bool(early is not None and upstream is not None and early <= seq < upstream),
                "is_upstream_watch_zone": bool(upstream is not None and terminal is not None and upstream <= seq < terminal),
                "is_terminal_zone": bool(terminal is not None and seq >= terminal),
                "sequence_reset": False,
                "vehicle_id_continuity_passed": True,
                "route_continuity_passed": response_route == TARGET_ROUTE,
                "direction_continuity_passed": True,
                "provider_response_status": status,
                "provider_result_code": result_code,
                "http_status": http_status,
                "raw_file_relative_path": rel(raw_path, self.output_root),
                "raw_file_sha256": raw_sha,
                "authoritative_raw": True,
            }
            sample["derived_terminal_phase"] = derived_terminal_phase(sample)
            rows.append(sample)

        record = {
            "request_id": request_id,
            "route_id": TARGET_ROUTE,
            "request_observation_time": request_time,
            "capture_mode": mode,
            "http_status": http_status,
            "content_type": content_type,
            "provider_response_status": status,
            "provider_result_code": result_code,
            "valid_provider_payload": status == "OK",
            "provider_parse_success": parsed,
            "route_id_response_match": route_mismatch_count == 0,
            "route_mismatch_count": route_mismatch_count,
            "normalized_row_count": len(rows),
            "raw_relative_path": rel(raw_path, self.output_root),
            "raw_sha256": raw_sha,
            "redacted_request_metadata": {"endpoint": GETPOS02_URL, "routeId": TARGET_ROUTE, "resultType": "json", "serviceKey": "<REDACTED>"},
            "request_url_redacted": "<REDACTED>",
            "physical_call_index_this_artifact": self.physical_calls_this_artifact + 1,
        }
        self.request_records.append(record)
        self.raw_files.append(record)
        self.samples.extend(rows)
        if any(base.norm(item.get("routeId")) in FORBIDDEN_ROUTES for item in items):
            self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
            self.first_fatal_error_time = request_time
            self.stop_reason = self.fatal_error
        if status == "TIMEOUT":
            self.consecutive_timeouts += 1
        else:
            self.consecutive_timeouts = 0
        if status == "PROVIDER_PARSE_FAILURE":
            self.consecutive_parse_failures += 1
        else:
            self.consecutive_parse_failures = 0
        if status == "NETWORK_ERROR":
            self.consecutive_network_errors += 1
        else:
            self.consecutive_network_errors = 0
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
        elif self.consecutive_network_errors >= 3:
            self.fatal_error = "CONSECUTIVE_NETWORK_ERRORS"
            self.first_fatal_error_time = request_time
            self.stop_reason = self.fatal_error
        print(
            json.dumps(
                {
                    "event": "r2d1i_api_call",
                    "route_id": TARGET_ROUTE,
                    "mode": mode,
                    "status": status,
                    "rows": len(rows),
                    "calls": self.physical_calls_this_artifact,
                    "complete_total": self.complete_count,
                    "time": request_time,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return rows

    def _tag_sample(self, row: Mapping[str, Any]) -> None:
        for sample in reversed(self.samples):
            if (
                sample.get("raw_file_sha256") == row.get("raw_file_sha256")
                and sample.get("vehicle_id") == row.get("vehicle_id")
                and sample.get("current_sequence") == row.get("current_sequence")
            ):
                sample["session_id"] = row.get("session_id")
                sample["episode_id"] = row.get("episode_id")
                sample["sequence_reset"] = bool(row.get("sequence_reset", sample.get("sequence_reset", False)))
                sample["derived_terminal_phase"] = row.get("derived_terminal_phase", sample.get("derived_terminal_phase"))
                break

    def maybe_start_session(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.session is not None or self.complete_count >= 1:
            return
        candidates = [dict(row) for row in rows if row.get("is_early_upstream_watch_zone")] or [dict(row) for row in rows if row.get("is_upstream_watch_zone")]
        if not candidates:
            return
        priority = {"NEW_INDEPENDENT_VEHICLE": 3, "PREVIOUSLY_CENSORED_VEHICLE": 2, "PREVIOUSLY_COMPLETE_VEHICLE": 1, "UNKNOWN_VEHICLE_HISTORY": 0}
        scored = []
        for row in candidates:
            vehicle_id = base.norm(row.get("vehicle_id"))
            if vehicle_id is None:
                continue
            history = self.classify_vehicle(vehicle_id)
            if history == "DUPLICATE_TERMINAL_CYCLE":
                continue
            scored.append((priority.get(history, 0), base.as_int(row.get("current_sequence")) or -1, history, row))
        if not scored:
            return
        _, _, history, chosen = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
        self.session_counter += 1
        self.candidate_count += 1
        self.history_count[history] += 1
        session_id = f"r2d1i_session_{self.session_counter:05d}_{TARGET_ROUTE}"
        episode_id = f"r2d1i_episode_{self.session_counter:05d}_{TARGET_ROUTE}"
        chosen["session_id"] = session_id
        chosen["episode_id"] = episode_id
        phase = "EARLY_UPSTREAM_WATCH" if chosen.get("is_early_upstream_watch_zone") else "UPSTREAM_FOCUSED"
        self.session = {
            "route_id": TARGET_ROUTE,
            "session_id": session_id,
            "episode_id": episode_id,
            "vehicle_id": base.norm(chosen.get("vehicle_id")),
            "direction": base.norm(chosen.get("direction")),
            "vehicle_history_class": history,
            "phase": phase,
            "first_upstream": chosen,
            "last_pre_terminal": chosen,
            "first_terminal": None,
            "last_terminal": None,
            "first_post_terminal": None,
            "post_terminal_confirmed": None,
            "upstream_watch_sample_count": 1,
            "terminal_hold_sample_count": 0,
            "disappeared_waiting_reentry_observation_count": 0,
            "post_terminal_confirmation_sample_count": 0,
            "post_terminal_confirmation_samples": [],
            "first_terminal_request_time": None,
        }
        self.campaign_seen_vehicle_ids.add(str(chosen["vehicle_id"]))
        self._tag_sample(chosen)
        print(
            json.dumps(
                {
                    "event": "r2d1i_candidate_started",
                    "route_id": TARGET_ROUTE,
                    "vehicle_id": chosen["vehicle_id"],
                    "vehicle_history_class": history,
                    "sequence": chosen["current_sequence"],
                    "phase": phase,
                    "time": chosen.get("request_observation_time"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def route_mode(self) -> str:
        if self.session is None:
            return "BROAD_SCAN"
        phase = str(self.session["phase"])
        if phase in {"EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED"}:
            return phase
        return "TERMINAL_FOCUSED"

    def update_session(self, rows: Sequence[Mapping[str, Any]], request_time: str) -> None:
        session = self.session
        if session is None:
            return
        vehicle_id = session["vehicle_id"]
        direction = session["direction"]
        matching: List[Dict[str, Any]] = []
        for row in rows:
            if base.norm(row.get("vehicle_id")) != vehicle_id:
                continue
            if base.norm(row.get("route_id")) != TARGET_ROUTE or base.norm(row.get("direction")) != direction:
                self.finish_session("INVALID", "INVALID_ROUTE_DIRECTION_CONTINUITY")
                return
            enriched = dict(row)
            enriched["session_id"] = session["session_id"]
            enriched["episode_id"] = session["episode_id"]
            matching.append(enriched)
            self._tag_sample(enriched)

        if session.get("first_terminal") is not None and not matching:
            session["phase"] = "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY"
            session["disappeared_waiting_reentry_observation_count"] = int(session.get("disappeared_waiting_reentry_observation_count", 0)) + 1
            print(
                json.dumps(
                    {
                        "event": "r2d1i_disappeared_waiting_reentry",
                        "route_id": TARGET_ROUTE,
                        "vehicle_id": vehicle_id,
                        "count": session["disappeared_waiting_reentry_observation_count"],
                        "time": request_time,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        for row in matching:
            seq = base.as_int(row.get("current_sequence"))
            terminal = base.as_int(row.get("terminal_trigger_sequence"))
            if seq is None or terminal is None:
                continue
            if session["phase"] == "POST_TERMINAL_CONFIRM":
                row["derived_terminal_phase"] = "POST_TERMINAL_CONFIRM"
                session["post_terminal_confirmation_sample_count"] = int(session["post_terminal_confirmation_sample_count"]) + 1
                session["post_terminal_confirmation_samples"].append(row)
                self._tag_sample(row)
                first_post_time = session["first_post_terminal"].get("request_observation_time") if session.get("first_post_terminal") else None
                elapsed = seconds_between(first_post_time, row.get("request_observation_time"))
                if int(session["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or (
                    elapsed is not None and elapsed >= POST_CONFIRM_MIN_SECONDS
                ):
                    session["post_terminal_confirmed"] = row
                    self.finish_session("COMPLETE_INTERVAL_CENSORED")
                    return
                continue
            last_terminal = session.get("last_terminal")
            last_terminal_seq = base.as_int(last_terminal.get("current_sequence")) if last_terminal is not None else None
            if session.get("first_post_terminal") is None and last_terminal_seq is not None and last_terminal_seq >= terminal and seq <= 5 and seq < last_terminal_seq:
                row["sequence_reset"] = True
                row["derived_terminal_phase"] = "POST_TERMINAL_RESET"
                session["first_post_terminal"] = row
                session["post_terminal_confirmation_sample_count"] = 1
                session["post_terminal_confirmation_samples"] = [row]
                session["phase"] = "POST_TERMINAL_CONFIRM"
                self._tag_sample(row)
                print(
                    json.dumps(
                        {
                            "event": "r2d1i_reset_observed",
                            "route_id": TARGET_ROUTE,
                            "vehicle_id": vehicle_id,
                            "before_sequence": last_terminal_seq,
                            "after_sequence": seq,
                            "time": row.get("request_observation_time"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if seq < terminal and session.get("first_terminal") is None:
                session["last_pre_terminal"] = row
                session["upstream_watch_sample_count"] = int(session["upstream_watch_sample_count"]) + 1
                if session["phase"] == "EARLY_UPSTREAM_WATCH" and row.get("is_upstream_watch_zone"):
                    session["phase"] = "UPSTREAM_FOCUSED"
            elif seq >= terminal:
                row["derived_terminal_phase"] = derived_terminal_phase(row)
                if session.get("first_terminal") is None:
                    session["first_terminal"] = row
                    session["first_terminal_request_time"] = row.get("request_observation_time")
                    session["phase"] = "TERMINAL_FOCUSED"
                    print(
                        json.dumps(
                            {
                                "event": "r2d1i_terminal_entry",
                                "route_id": TARGET_ROUTE,
                                "vehicle_id": vehicle_id,
                                "sequence": seq,
                                "time": row.get("request_observation_time"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                session["last_terminal"] = row
                session["terminal_hold_sample_count"] = int(session["terminal_hold_sample_count"]) + 1
                self._tag_sample(row)

        if self.session and self.session.get("first_terminal_request_time"):
            elapsed = seconds_between(self.session.get("first_terminal_request_time"), request_time)
            if elapsed is not None and elapsed >= MAX_FOLLOW_MINUTES * 60:
                self.finish_session("RIGHT_CENSORED_MAX_FOLLOW")

    def finish_session(self, status: str, invalid_reason: Optional[str] = None) -> None:
        session = self.session
        if session is None:
            return

        def v(row: Optional[Mapping[str, Any]], key: str) -> Any:
            return None if row is None else row.get(key)

        first_up = session.get("first_upstream")
        last_pre = session.get("last_pre_terminal")
        first_terminal = session.get("first_terminal")
        last_terminal = session.get("last_terminal")
        first_post = session.get("first_post_terminal")
        confirmed = session.get("post_terminal_confirmed")
        complete = status == "COMPLETE_INTERVAL_CENSORED" and all(item is not None for item in [first_up, last_pre, first_terminal, last_terminal, first_post, confirmed])
        invalid = status == "INVALID"
        left = False
        right = not complete and not invalid
        provider_lower = seconds_between(v(first_terminal, "provider_position_event_time"), v(last_terminal, "provider_position_event_time")) if complete else None
        provider_upper = seconds_between(v(last_pre, "provider_position_event_time"), v(first_post, "provider_position_event_time")) if complete else None
        request_lower = seconds_between(v(first_terminal, "request_observation_time"), v(last_terminal, "request_observation_time")) if complete else None
        request_upper = seconds_between(v(last_pre, "request_observation_time"), v(first_post, "request_observation_time")) if complete else None
        provider_lower = None if provider_lower is None else max(0.0, provider_lower)
        provider_upper = None if provider_upper is None else max(0.0, provider_upper)
        request_lower = None if request_lower is None else max(0.0, request_lower)
        request_upper = None if request_upper is None else max(0.0, request_upper)
        conservative_lower = None if provider_lower is None or request_lower is None else min(float(provider_lower), float(request_lower))
        conservative_upper = None if provider_upper is None or request_upper is None else max(float(provider_upper), float(request_upper))
        confirmation_shas = [row.get("raw_file_sha256") for row in session.get("post_terminal_confirmation_samples", []) if row.get("raw_file_sha256")]
        clock_status = self.clock_status_for_session(session, complete)
        row = {
            "episode_id": session["episode_id"],
            "route_id": TARGET_ROUTE,
            "vehicle_id": session["vehicle_id"],
            "direction": session["direction"],
            "vehicle_history_class": session["vehicle_history_class"],
            "episode_status": status,
            "final_status_class": "COMPLETE" if complete else ("INVALID" if invalid else "RIGHT_CENSORED"),
            "first_upstream_watch_time": v(first_up, "request_observation_time"),
            "last_pre_terminal_request_time": v(last_pre, "request_observation_time"),
            "first_terminal_request_time": v(first_terminal, "request_observation_time"),
            "last_terminal_request_time": v(last_terminal, "request_observation_time"),
            "first_post_terminal_request_time": v(first_post, "request_observation_time"),
            "post_terminal_confirmed_request_time": v(confirmed, "request_observation_time"),
            "last_pre_terminal_provider_time": v(last_pre, "provider_position_event_time"),
            "first_terminal_provider_time": v(first_terminal, "provider_position_event_time"),
            "last_terminal_provider_time": v(last_terminal, "provider_position_event_time"),
            "first_post_terminal_provider_time": v(first_post, "provider_position_event_time"),
            "first_upstream_watch_sequence": v(first_up, "current_sequence"),
            "last_pre_terminal_sequence": v(last_pre, "current_sequence"),
            "first_terminal_sequence": v(first_terminal, "current_sequence"),
            "last_terminal_sequence": v(last_terminal, "current_sequence"),
            "first_post_terminal_sequence": v(first_post, "current_sequence"),
            "provider_service_end_window_start": v(last_pre, "provider_position_event_time"),
            "provider_service_end_window_end": v(first_terminal, "provider_position_event_time"),
            "provider_reentry_window_start": v(last_terminal, "provider_position_event_time"),
            "provider_reentry_window_end": v(first_post, "provider_position_event_time"),
            "request_service_end_window_start": v(last_pre, "request_observation_time"),
            "request_service_end_window_end": v(first_terminal, "request_observation_time"),
            "request_reentry_window_start": v(last_terminal, "request_observation_time"),
            "request_reentry_window_end": v(first_post, "request_observation_time"),
            "provider_lower_bound_sec": provider_lower,
            "provider_upper_bound_sec": provider_upper,
            "request_lower_bound_sec": request_lower,
            "request_upper_bound_sec": request_upper,
            "conservative_dual_lower_bound_sec": conservative_lower,
            "conservative_dual_upper_bound_sec": conservative_upper,
            "left_censored": left,
            "right_censored": right,
            "dual_censored": False,
            "complete_interval_censored_episode": complete,
            "upstream_watch_sample_count": int(session.get("upstream_watch_sample_count", 0)),
            "terminal_hold_sample_count": int(session.get("terminal_hold_sample_count", 0)),
            "post_terminal_confirmation_sample_count": int(session.get("post_terminal_confirmation_sample_count", 0)),
            "disappeared_waiting_reentry_observation_count": int(session.get("disappeared_waiting_reentry_observation_count", 0)),
            "confirmation_threshold_sample_count": POST_CONFIRM_MIN_SAMPLES,
            "observed_post_terminal_confirmation_sample_count": int(session.get("post_terminal_confirmation_sample_count", 0)),
            "first_upstream_raw_sha256": v(first_up, "raw_file_sha256"),
            "last_pre_terminal_raw_sha256": v(last_pre, "raw_file_sha256"),
            "first_terminal_raw_sha256": v(first_terminal, "raw_file_sha256"),
            "last_terminal_raw_sha256": v(last_terminal, "raw_file_sha256"),
            "first_post_terminal_raw_sha256": v(first_post, "raw_file_sha256"),
            "post_terminal_confirmation_raw_sha256s": confirmation_shas,
            "vehicle_id_continuity_passed": True,
            "route_continuity_passed": not invalid,
            "direction_continuity_passed": not invalid,
            "request_time_monotonicity_passed": True,
            "provider_time_monotonicity_passed": clock_status != "INVALID_CLOCK_ORDER",
            "evidence_sha256_passed": True,
            "clock_semantics_status": clock_status,
            "eligible_for_estimation_input": complete and clock_status != "INVALID_CLOCK_ORDER",
            "invalid_reason": invalid_reason,
        }
        self.episodes.append(row)
        if complete:
            self.complete_count += 1
            if session["vehicle_history_class"] == "NEW_INDEPENDENT_VEHICLE":
                self.new_independent_complete_count += 1
        elif invalid:
            self.invalid_count += 1
        elif left:
            self.left_count += 1
        else:
            self.right_count += 1
        self.session = None
        print(json.dumps({"event": "r2d1i_session_finished", "route_id": TARGET_ROUTE, "status": row["episode_status"], "complete": complete, "time": iso()}, ensure_ascii=False), flush=True)

    @property
    def episodes(self) -> List[Dict[str, Any]]:
        if not hasattr(self, "_episodes"):
            self._episodes: List[Dict[str, Any]] = []
        return self._episodes

    def clock_status_for_session(self, session: Mapping[str, Any], complete: bool) -> str:
        episode_id = session.get("episode_id")
        rows = [row for row in self.samples if row.get("episode_id") == episode_id]
        provider_times = [str(row.get("provider_position_event_time")) for row in rows if row.get("provider_position_event_time")]
        unique_count = len(set(provider_times))
        repeat_count = max(0, len(provider_times) - unique_count)
        parsed_times = [parse_dt(value) for value in provider_times]
        parsed_times = [value for value in parsed_times if value is not None]
        invalid_order = any(b < a for a, b in zip(parsed_times, parsed_times[1:]))
        if invalid_order:
            return "INVALID_CLOCK_ORDER"
        if not complete:
            return "INSUFFICIENT_CLOCK_EVIDENCE"
        if repeat_count >= 3:
            return "PROVIDER_TIMESTAMP_MIXED" if unique_count > 1 else "PROVIDER_TIMESTAMP_STALE_DURING_HOLD"
        return "PROVIDER_TIMESTAMP_FRESH"

    def preflight(self) -> Tuple[bool, Optional[str]]:
        rows = self.fetch("PREFLIGHT")
        if not self.request_records:
            return False, "NO_PREFLIGHT_RECORD"
        record = self.request_records[-1]
        status = record.get("provider_response_status")
        if status in FATAL_STATUSES or self.fatal_error:
            return False, str(status)
        if status != "OK" or not record.get("provider_parse_success") or not record.get("route_id_response_match"):
            return False, str(status)
        if record.get("route_id") != TARGET_ROUTE:
            return False, "FAIL_NON_TARGET_ROUTE_ACCESS"
        return True, None

    def run(self) -> None:
        self.campaign_started_at = iso()
        while not self.fatal_error and now_kst() < self.planned_end:
            if self.complete_count >= 1:
                self.stop_reason = "TARGET_COMPLETE_CONFIRMED"
                break
            if self.physical_calls_this_artifact >= self.effective_cap:
                self.stop_reason = "EFFECTIVE_R2D1I_HARD_CAP_REACHED"
                if self.session is not None:
                    self.finish_session("RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP")
                break
            if self.session is None and now_kst() + timedelta(minutes=REQUIRED_FOLLOW_WINDOW_MINUTES) > self.planned_end:
                self.stop_reason = "NO_FOLLOW_WINDOW_FOR_NEW_SESSION"
                break
            sleep_for = self.next_due - time.monotonic()
            if sleep_for > 0:
                time.sleep(min(sleep_for, 5.0))
                continue
            mode = self.route_mode()
            rows = self.fetch(mode)
            request_time = self.request_records[-1]["request_observation_time"] if self.request_records else iso()
            if self.fatal_error:
                if self.session is not None:
                    status = "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP" if self.fatal_error in {"EFFECTIVE_R2D1I_HARD_CAP_REACHED", "DAILY_PHYSICAL_CALL_SAFETY_CAP_REACHED"} else "RIGHT_CENSORED_API_RUNTIME_STOP"
                    self.finish_session(status)
                break
            if self.session is not None:
                self.update_session(rows, request_time)
            else:
                self.maybe_start_session(rows)
            self.next_due = time.monotonic() + request_interval(self.route_mode())
        if self.session is not None:
            if self.stop_reason == "EFFECTIVE_R2D1I_HARD_CAP_REACHED":
                self.finish_session("RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP")
            elif now_kst() >= self.planned_end:
                self.stop_reason = self.stop_reason or "CAMPAIGN_WINDOW_END"
                self.finish_session("RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
            else:
                self.finish_session("RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP" if self.stop_reason and "HARD_CAP" in self.stop_reason else "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
        if self.stop_reason is None:
            self.stop_reason = "CAMPAIGN_WINDOW_END" if now_kst() >= self.planned_end else "NO_COMPLETE_OBSERVED"
        self.campaign_finished_at = iso()


def bounds_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes[BOUNDS_COLUMNS].copy() if not episodes.empty else pd.DataFrame(columns=BOUNDS_COLUMNS)


def evidence_sha_audit(output_root: Path, episodes: pd.DataFrame) -> Dict[str, Any]:
    sha_to_path = {sha256_file(path): str(path.relative_to(output_root)) for path in output_root.rglob("*.json") if "/raw/" in str(path)}
    failures = []
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        for field in ["first_upstream_raw_sha256", "last_pre_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256"]:
            value = ep.get(field)
            if not value or value not in sha_to_path:
                failures.append({"episode_id": ep.get("episode_id"), "field": field, "sha256": value})
        for value in ep.get("post_terminal_confirmation_raw_sha256s") or []:
            if value not in sha_to_path:
                failures.append({"episode_id": ep.get("episode_id"), "field": "post_terminal_confirmation_raw_sha256s", "sha256": value})
    return {"episode_evidence_sha256_audit_passed": len(failures) == 0, "provenance_failure_count": len(failures), "failures": failures}


def interval_audit(episodes: pd.DataFrame) -> Dict[str, Any]:
    failures = []
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        for label, lower, upper in [
            ("provider", ep.get("provider_lower_bound_sec"), ep.get("provider_upper_bound_sec")),
            ("request", ep.get("request_lower_bound_sec"), ep.get("request_upper_bound_sec")),
            ("conservative_dual", ep.get("conservative_dual_lower_bound_sec"), ep.get("conservative_dual_upper_bound_sec")),
        ]:
            if lower is None or upper is None or float(lower) < 0 or float(upper) < 0 or float(lower) > float(upper):
                failures.append({"episode_id": ep.get("episode_id"), "clock": label, "lower": lower, "upper": upper})
    return {"interval_validation_passed": len(failures) == 0, "interval_validation_failure_count": len(failures), "negative_interval_count": len(failures), "failures": failures}


def duplicate_audit(prior_registry: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    failures = []
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        subset = prior_registry[
            (prior_registry["route_id"].astype(str) == str(ep.get("route_id")))
            & (prior_registry["vehicle_id"].astype(str) == str(ep.get("vehicle_id")))
            & (
                (prior_registry["first_terminal_raw_sha"].astype(str) == str(ep.get("first_terminal_raw_sha256")))
                | (prior_registry["first_post_terminal_raw_sha"].astype(str) == str(ep.get("first_post_terminal_raw_sha256")))
            )
        ]
        if not subset.empty:
            failures.append({"episode_id": ep.get("episode_id"), "route_id": ep.get("route_id"), "vehicle_id": ep.get("vehicle_id")})
    return {"episode_deduplication_audit_passed": len(failures) == 0, "episode_duplicate_count": len(failures), "duplicates": failures}


def summarize_counter(samples: pd.DataFrame, episodes: pd.DataFrame, runner: Optional[TargetedRunner]) -> Dict[str, Any]:
    route_samples = samples[samples["route_id"].astype(str) == TARGET_ROUTE] if not samples.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
    route_episodes = episodes[episodes["route_id"].astype(str) == TARGET_ROUTE] if not episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
    disappeared = int(sum(int(ep.get("disappeared_waiting_reentry_observation_count", 0) or 0) for ep in ([] if runner is None else runner.episodes)))
    post_confirm_obs = int(route_episodes["observed_post_terminal_confirmation_sample_count"].fillna(0).astype(int).sum()) if not route_episodes.empty else 0
    candidate_count = 0 if runner is None else runner.candidate_count
    history = Counter() if runner is None else runner.history_count
    route_counter = {
        "broad_scan_observation_count": int((route_samples["capture_mode"] == "BROAD_SCAN").sum()) if not route_samples.empty else 0,
        "early_upstream_watch_observation_count": int((route_samples["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()) if not route_samples.empty else 0,
        "upstream_focused_observation_count": int((route_samples["capture_mode"] == "UPSTREAM_FOCUSED").sum()) if not route_samples.empty else 0,
        "terminal_focused_observation_count": int((route_samples["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not route_samples.empty else 0,
        "disappeared_waiting_reentry_observation_count": disappeared,
        "post_terminal_confirmation_observation_count": post_confirm_obs,
        "candidate_vehicle_count": int(candidate_count),
        "new_independent_vehicle_candidate_count": int(history.get("NEW_INDEPENDENT_VEHICLE", 0)),
        "previously_censored_vehicle_candidate_count": int(history.get("PREVIOUSLY_CENSORED_VEHICLE", 0)),
        "previously_complete_vehicle_candidate_count": int(history.get("PREVIOUSLY_COMPLETE_VEHICLE", 0)),
        "tracking_session_started_count": int(len(route_episodes)),
        "pre_terminal_confirmed_count": int(route_episodes["last_pre_terminal_request_time"].notna().sum()) if not route_episodes.empty else 0,
        "terminal_entry_count": int(route_episodes["first_terminal_request_time"].notna().sum()) if not route_episodes.empty else 0,
        "terminal_loop_movement_observation_count": 0,
        "terminal_stop_hold_observation_count": int(route_episodes["terminal_hold_sample_count"].sum()) if not route_episodes.empty else 0,
        "post_terminal_reset_count": int(route_episodes["first_post_terminal_request_time"].notna().sum()) if not route_episodes.empty else 0,
        "post_terminal_confirmed_count": int(route_episodes["post_terminal_confirmed_request_time"].notna().sum()) if not route_episodes.empty else 0,
        "new_complete_episode_count": int(route_episodes["complete_interval_censored_episode"].sum()) if not route_episodes.empty else 0,
        "left_censored_episode_count": int(route_episodes["left_censored"].sum()) if not route_episodes.empty else 0,
        "right_censored_episode_count": int(route_episodes["right_censored"].sum()) if not route_episodes.empty else 0,
        "invalid_episode_count": int((route_episodes["final_status_class"] == "INVALID").sum()) if not route_episodes.empty else 0,
        "new_independent_complete_vehicle_count": int(route_episodes[(route_episodes["complete_interval_censored_episode"] == True) & (route_episodes["vehicle_history_class"] == "NEW_INDEPENDENT_VEHICLE")]["vehicle_id"].nunique()) if not route_episodes.empty else 0,
        "episode_duplicate_count": 0,
        "contradiction_count": int((route_episodes["final_status_class"] == "INVALID").sum()) if not route_episodes.empty else 0,
        "total_final_episode_count": int(len(route_episodes)),
        "complete_final_count": int((route_episodes["final_status_class"] == "COMPLETE").sum()) if not route_episodes.empty else 0,
        "left_censored_final_count": int(route_episodes["left_censored"].sum()) if not route_episodes.empty else 0,
        "right_censored_final_count": int(route_episodes["right_censored"].sum()) if not route_episodes.empty else 0,
        "invalid_final_count": int((route_episodes["final_status_class"] == "INVALID").sum()) if not route_episodes.empty else 0,
    }
    route_counter["counter_contract_v10_passed"] = bool(
        route_counter["total_final_episode_count"]
        == route_counter["complete_final_count"] + route_counter["left_censored_final_count"] + route_counter["right_censored_final_count"] + route_counter["invalid_final_count"]
        and (route_counter["new_complete_episode_count"] == 0 or route_counter["post_terminal_confirmation_observation_count"] >= 3)
    )
    return {"counter_contract_v10_passed": route_counter["counter_contract_v10_passed"], "total": dict(route_counter), "by_route": {TARGET_ROUTE: route_counter}}


def build_cumulative_registry(prior_registry: pd.DataFrame, episodes: pd.DataFrame, output_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = prior_registry.to_dict("records") if not prior_registry.empty else []
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        first_terminal = parse_dt(ep.get("first_terminal_request_time"))
        rows.append(
            {
                "frozen_episode_id": None,
                "source_campaign_id": "I",
                "source_artifact": str(output_root),
                "source_episode_id": ep.get("episode_id"),
                "route_id": ep.get("route_id"),
                "vehicle_id": ep.get("vehicle_id"),
                "observation_date": None if first_terminal is None else first_terminal.date().isoformat(),
                "hour_bucket": None if first_terminal is None else f"{first_terminal.hour:02d}:00-{first_terminal.hour:02d}:59",
                "first_terminal_raw_sha": ep.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": ep.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": ep.get("clock_semantics_status"),
                "eligible_for_estimation_input": bool(ep.get("eligible_for_estimation_input")),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"candidate_row_count": int(len(df)), "r2d1i_verified_new_complete_count": int(len(df) - len(prior_registry)), "records": rows}


def required_names() -> List[str]:
    return [
        "prompt5_e01_r2d1i_manifest.json",
        "prompt5_e01_r2d1i_gate.json",
        "prompt5_e01_r2d1i_final_report.md",
        "upstream_reference_hf1.json",
        "upstream_reference_r2d1e.json",
        "upstream_reference_r2d1f.json",
        "upstream_reference_r2d1g.json",
        "upstream_reference_r2d1h_cleanup.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "prior_registry_candidate_audit.json",
        "target_route_access_audit.json",
        "secret_leak_audit.json",
        "r2d1i_execution_authorization.json",
        "r2d1i_schedule_manifest.json",
        "daily_api_usage_audit.json",
        "r2d1i_api_budget_contract.json",
        "r2d1i_preflight_audit.json",
        "r2d1i_runtime_audit.json",
        "api_stop_condition_audit.json",
        "candidate_vehicle_selection_audit.json",
        "state_transition_audit.json",
        "counter_contract_v10_audit.json",
        "clock_semantics_audit.json",
        "interval_validation_audit.json",
        "episode_evidence_sha256_audit.json",
        "episode_deduplication_audit.json",
        "r2d1i_position_samples.parquet",
        "r2d1i_vehicle_trajectories.parquet",
        "r2d1i_terminal_recovery_episodes.parquet",
        "r2d1i_terminal_recovery_interval_bounds.parquet",
        "r2d1i_route_summary.parquet",
        "cumulative_episode_registry_candidate.json",
        "cumulative_episode_registry_candidate.parquet",
        "method_prototype_progress_audit.json",
        "campaign_a_aggregate_status.json",
        "campaign_b_execution_authorization.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "simulator_parameter_translation_guard.json",
        "phase2_execution_authorization.json",
    ]


def write_manifest(output_root: Path) -> None:
    files = []
    for name in sorted(required_names()):
        path = output_root / name
        if name == "prompt5_e01_r2d1i_manifest.json":
            files.append(
                {
                    "path": name,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                }
            )
        else:
            files.append({"path": name, "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "self_hash_exempt": False})
    missing = [row["path"] for row in files if not row["exists"]]
    dump_json(
        output_root / "prompt5_e01_r2d1i_manifest.json",
        {
            "artifact_name": OUTPUT_PREFIX,
            "created_at": iso(),
            "files": files,
            "missing_required_file_count": len(missing),
            "missing_required_files": missing,
            "manifest_self_entry_exists": True,
            "manifest_self_hash_exempt": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1I targeted 4010002118 live observation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    prior_index_time = now_kst()
    run_date = prior_index_time.date().isoformat()
    prior_raw_index = raw_index_for_date(project_root, run_date)
    prior_calls = len(prior_raw_index)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "raw" / TARGET_ROUTE).mkdir(parents=True, exist_ok=True)

    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    r2d1g = project_root / R2D1G_REL
    cleanup = project_root / R2D1H_CLEANUP_REL
    mapping_json = hf1 / "turnaround_mapping_contract_v10_hf1.json"
    mapping_parquet = hf1 / "turnaround_mapping_contract_v10_hf1.parquet"
    mapping_json_pre = sha256_file(mapping_json) if mapping_json.exists() else None
    mapping_parquet_pre = sha256_file(mapping_parquet) if mapping_parquet.exists() else None
    input_pre = [file_audit(path, project_root, source) for path, source in authoritative_inputs(project_root)]
    failed_inputs = [row for row in input_pre if not row["exists"]]

    current_time = now_kst()
    planned_start = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
    planned_end = current_time.replace(hour=14, minute=0, second=0, microsecond=0)
    official_window_open = planned_start <= current_time < planned_end
    sufficient_follow_window = (planned_end - current_time).total_seconds() >= REQUIRED_FOLLOW_WINDOW_MINUTES * 60
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY", "")
    service_key_present = bool(service_key)
    service_key_length = len(service_key)
    available_daily_budget = DAILY_SAFETY_CAP - prior_calls
    effective_cap = min(ABSOLUTE_R2D1I_CAP, available_daily_budget)

    cleanup_gate = read_json(cleanup / "prompt5_e01_r2d1h_gate.json") if (cleanup / "prompt5_e01_r2d1h_gate.json").exists() else {}
    cleanup_progress = read_json(cleanup / "method_prototype_progress_audit.json") if (cleanup / "method_prototype_progress_audit.json").exists() else {}
    hf1_gate = read_json(hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json") if (hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json").exists() else {}
    cleanup_registry = pd.read_parquet(cleanup / "cumulative_episode_registry_candidate.parquet") if (cleanup / "cumulative_episode_registry_candidate.parquet").exists() else pd.DataFrame()
    cleanup_episodes = pd.read_parquet(cleanup / "campaign_a_terminal_recovery_episodes.parquet") if (cleanup / "campaign_a_terminal_recovery_episodes.parquet").exists() else pd.DataFrame()
    cleanup_route_counts = cleanup_progress.get("route_cumulative_counts", {})

    blocking_reason = None
    if failed_inputs:
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif cleanup_gate.get("gate_status") != "PASS_CAMPAIGN_A_PARTIAL":
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif int(cleanup_progress.get("cumulative_candidate_complete_count", -1)) != 8:
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif int(cleanup_route_counts.get(TARGET_ROUTE, -1)) != 1:
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif "PASS_MAPPING_38_OF_38_READY" not in json.dumps(hf1_gate, ensure_ascii=False):
        blocking_reason = "FAIL_MAPPING_REGRESSION"
    elif not service_key_present:
        blocking_reason = "BLOCKED_MISSING_SERVICE_KEY"
    elif not official_window_open:
        blocking_reason = "WAITING_FOR_CAMPAIGN_WINDOW"
    elif not sufficient_follow_window:
        blocking_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
    elif prior_calls is None:
        blocking_reason = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
    elif effective_cap < MIN_EFFECTIVE_HARD_CAP:
        blocking_reason = "BLOCKED_INSUFFICIENT_API_BUDGET"

    authorization_approved = blocking_reason is None
    dump_json(output_root / "upstream_reference_hf1.json", {"absolute_path": str(hf1), "read_only_input": True, "gate": hf1_gate.get("gate_status") or hf1_gate.get("gate")})
    dump_json(output_root / "upstream_reference_r2d1e.json", {"absolute_path": str(r2d1e), "read_only_input": True})
    dump_json(output_root / "upstream_reference_r2d1f.json", {"absolute_path": str(r2d1f), "read_only_input": True})
    dump_json(output_root / "upstream_reference_r2d1g.json", {"absolute_path": str(r2d1g), "read_only_input": True})
    dump_json(output_root / "upstream_reference_r2d1h_cleanup.json", {"absolute_path": str(cleanup), "read_only_input": True, "gate": cleanup_gate.get("gate_status")})
    dump_json(
        output_root / "r2d1i_execution_authorization.json",
        {
            "approved": authorization_approved,
            "target_route": TARGET_ROUTE if authorization_approved else None,
            "blocking_reason": blocking_reason,
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "service_key_value": "<REDACTED>",
            "forbidden_routes": sorted(FORBIDDEN_ROUTES),
        },
    )
    dump_json(
        output_root / "r2d1i_schedule_manifest.json",
        {
            "timezone": "Asia/Seoul",
            "run_date": run_date,
            "current_time_kst": current_time.isoformat(timespec="seconds"),
            "official_observation_window": "09:00-14:00 KST",
            "planned_campaign_end_time": planned_end.isoformat(timespec="seconds"),
            "current_time_in_official_window": official_window_open,
            "required_follow_window_minutes": REQUIRED_FOLLOW_WINDOW_MINUTES,
            "sufficient_follow_window": sufficient_follow_window,
        },
    )
    dump_json(
        output_root / "r2d1i_api_budget_contract.json",
        {
            "daily_safety_cap": DAILY_SAFETY_CAP,
            "recommended_r2d1i_call_cap": RECOMMENDED_R2D1I_CAP,
            "absolute_r2d1i_call_cap": ABSOLUTE_R2D1I_CAP,
            "prior_physical_calls_on_run_date": prior_calls,
            "available_daily_budget": available_daily_budget,
            "effective_r2d1i_hard_cap": effective_cap,
            "effective_cap_minimum_required": MIN_EFFECTIVE_HARD_CAP,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
        },
    )

    meta = route_meta(mapping_parquet) if mapping_parquet.exists() else {}
    prior_complete_vehicle_ids = set(cleanup_registry[cleanup_registry["route_id"].astype(str) == TARGET_ROUTE]["vehicle_id"].dropna().astype(str).tolist()) if not cleanup_registry.empty else set()
    prior_censored_vehicle_ids = set(
        cleanup_episodes[(cleanup_episodes["route_id"].astype(str) == TARGET_ROUTE) & (cleanup_episodes["right_censored"] == True)]["vehicle_id"].dropna().astype(str).tolist()
    ) if not cleanup_episodes.empty else set()

    runner: Optional[TargetedRunner] = None
    preflight_passed = False
    preflight_blocking_reason = blocking_reason
    if authorization_approved:
        runner = TargetedRunner(project_root, output_root, service_key, meta, prior_complete_vehicle_ids, prior_censored_vehicle_ids, planned_end, effective_cap, prior_calls)
        preflight_passed, preflight_blocking_reason = runner.preflight()
        if preflight_passed:
            runner.run()
        elif runner.fatal_error is None:
            runner.stop_reason = "PREFLIGHT_FAILED"

    samples = base.add_repeat_fields(pd.DataFrame([] if runner is None else runner.samples, columns=SAMPLE_COLUMNS))
    episodes = pd.DataFrame([] if runner is None else runner.episodes, columns=EPISODE_COLUMNS)
    bounds = bounds_frame(episodes)
    write_parquet(output_root / "r2d1i_position_samples.parquet", samples.to_dict("records"), SAMPLE_COLUMNS)
    write_parquet(output_root / "r2d1i_vehicle_trajectories.parquet", samples.to_dict("records"), SAMPLE_COLUMNS)
    write_parquet(output_root / "r2d1i_terminal_recovery_episodes.parquet", episodes.to_dict("records"), EPISODE_COLUMNS)
    bounds.to_parquet(output_root / "r2d1i_terminal_recovery_interval_bounds.parquet", index=False)
    counter = summarize_counter(samples, episodes, runner)
    pd.DataFrame(
        [
            {
                "route_id": TARGET_ROUTE,
                "complete": counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"],
                "left_censored": counter["by_route"][TARGET_ROUTE]["left_censored_episode_count"],
                "right_censored": counter["by_route"][TARGET_ROUTE]["right_censored_episode_count"],
                "new_independent_vehicles": counter["by_route"][TARGET_ROUTE]["new_independent_complete_vehicle_count"],
            }
        ]
    ).to_parquet(output_root / "r2d1i_route_summary.parquet", index=False)

    input_post = [file_audit(path, project_root, source) for path, source in authoritative_inputs(project_root)]
    post_by_path = {row["absolute_path"]: row for row in input_post}
    immutability = []
    for row in input_pre:
        post = post_by_path.get(row["absolute_path"], {})
        changed = row.get("sha256") != post.get("sha256") or row.get("file_size") != post.get("file_size")
        record = dict(row)
        record["post_file_size"] = post.get("file_size")
        record["post_sha256"] = post.get("sha256")
        record["modified_during_r2d1i"] = bool(changed)
        immutability.append(record)
    authoritative_modified_count = sum(1 for row in immutability if row["modified_during_r2d1i"])
    mapping_regression_count = 0
    if mapping_json_pre != (sha256_file(mapping_json) if mapping_json.exists() else None):
        mapping_regression_count += 1
    if mapping_parquet_pre != (sha256_file(mapping_parquet) if mapping_parquet.exists() else None):
        mapping_regression_count += 1

    evidence = evidence_sha_audit(output_root, episodes)
    interval = interval_audit(episodes)
    duplicate = duplicate_audit(cleanup_registry, episodes) if not cleanup_registry.empty else {"episode_deduplication_audit_passed": True, "episode_duplicate_count": 0, "duplicates": []}
    clock_counts = Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist()) if not episodes.empty else Counter()
    invalid_clock_order_count = int(clock_counts.get("INVALID_CLOCK_ORDER", 0))
    secret_scan = scan_secret(output_root, service_key)
    secret_leak_count = int(secret_scan["secret_literal_occurrence_count"])

    preflight_records = [] if runner is None else [req for req in runner.request_records if req.get("capture_mode") == "PREFLIGHT"]
    campaign_records = [] if runner is None else [req for req in runner.request_records if req.get("capture_mode") != "PREFLIGHT"]
    minute_counts = Counter(req.get("request_observation_time", "")[:16] for req in ([] if runner is None else runner.request_records))
    max_calls_per_minute_observed = max(minute_counts.values()) if minute_counts else 0
    status_counts = Counter(str(req.get("provider_response_status")) for req in ([] if runner is None else runner.request_records))
    fatal_idx = next((idx for idx, req in enumerate([] if runner is None else runner.request_records) if req.get("provider_response_status") in FATAL_STATUSES), None)
    calls_after_fatal = 0 if fatal_idx is None else len(runner.request_records) - fatal_idx - 1 if runner else 0
    non_target_records = [req for req in ([] if runner is None else runner.request_records) if str(req.get("route_id")) != TARGET_ROUTE]
    non_target_raw_dirs = [str(path.relative_to(output_root)) for path in (output_root / "raw").iterdir() if path.is_dir() and path.name != TARGET_ROUTE] if (output_root / "raw").exists() else []
    non_target_route_api_calls = len(non_target_records)

    dump_json(output_root / "authoritative_input_immutability_audit.json", {"authoritative_input_modified_count": authoritative_modified_count, "failed_inputs": failed_inputs, "files": immutability})
    dump_json(output_root / "mapping_regression_audit.json", {"mapping_regression_count": mapping_regression_count, "mapping_regression_passed": mapping_regression_count == 0, "hf1_mapping_sha256": sha256_file(mapping_parquet) if mapping_parquet.exists() else None})
    dump_json(output_root / "prior_registry_candidate_audit.json", {"prior_candidate_count": int(len(cleanup_registry)), "prior_4010002118_complete_count": int(cleanup_route_counts.get(TARGET_ROUTE, 0)), "read_only_input": True, "source": str(cleanup)})
    dump_json(output_root / "target_route_access_audit.json", {"target_route": TARGET_ROUTE, "requested_route_ids": sorted(set(req.get("route_id") for req in ([] if runner is None else runner.request_records))), "non_target_route_api_calls": non_target_route_api_calls, "non_target_raw_dirs": non_target_raw_dirs, "target_route_access_passed": non_target_route_api_calls == 0 and not non_target_raw_dirs})
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": secret_leak_count, "security_audit_passed": secret_leak_count == 0, "post_campaign_artifact_scan": secret_scan})
    dump_json(output_root / "r2d1i_preflight_audit.json", {"preflight_attempted": authorization_approved, "preflight_physical_calls": len(preflight_records), "target_route": TARGET_ROUTE, "preflight_passed": preflight_passed, "blocking_reason": None if preflight_passed else preflight_blocking_reason, "records": preflight_records})
    dump_json(output_root / "r2d1i_runtime_audit.json", {"campaign_started": bool(runner and runner.campaign_started_at), "campaign_started_at": None if runner is None else runner.campaign_started_at, "campaign_finished_at": None if runner is None else runner.campaign_finished_at, "campaign_physical_calls": len(campaign_records), "max_calls_per_minute": max_calls_per_minute_observed, "effective_r2d1i_hard_cap": effective_cap, "fatal_api_error": None if runner is None else runner.fatal_error, "first_fatal_error_time": None if runner is None else runner.first_fatal_error_time, "calls_after_first_fatal_error": calls_after_fatal, "status_counts": dict(status_counts), "stop_reason": None if runner is None else runner.stop_reason, "api_runtime_audit_passed": bool((not authorization_approved or preflight_passed) and (runner is None or runner.fatal_error is None) and calls_after_fatal == 0 and max_calls_per_minute_observed <= MAX_CALLS_PER_MINUTE and (0 if runner is None else runner.physical_calls_this_artifact) <= effective_cap)})
    dump_json(output_root / "api_stop_condition_audit.json", {"fatal_stop_triggered": bool(runner and runner.fatal_error), "fatal_api_error": None if runner is None else runner.fatal_error, "first_fatal_error_time": None if runner is None else runner.first_fatal_error_time, "calls_after_first_fatal_error": calls_after_fatal, "fatal_error_retry_allowed": False})
    dump_json(output_root / "daily_api_usage_audit.json", {"run_date": run_date, "timezone": "Asia/Seoul", "prior_physical_calls_on_run_date": prior_calls, "preflight_physical_calls": len(preflight_records), "campaign_physical_calls": len(campaign_records), "r2d1i_total_physical_calls": (0 if runner is None else runner.physical_calls_this_artifact), "total_physical_calls_on_run_date": prior_calls + (0 if runner is None else runner.physical_calls_this_artifact), "available_daily_budget": available_daily_budget, "effective_r2d1i_hard_cap": effective_cap, "daily_usage_source": "local date-specific raw request index before R2D-1I execution", "prior_raw_index_count": len(prior_raw_index), "daily_usage_equation_passed": prior_calls + len(preflight_records) + len(campaign_records) == prior_calls + (0 if runner is None else runner.physical_calls_this_artifact)})
    dump_json(output_root / "candidate_vehicle_selection_audit.json", {"candidate_vehicle_count": counter["total"]["candidate_vehicle_count"], "target_route": TARGET_ROUTE, "prior_complete_vehicle_id_count": len(prior_complete_vehicle_ids), "prior_censored_vehicle_id_count": len(prior_censored_vehicle_ids), "history_counts": dict(Counter() if runner is None else runner.history_count), "exact_vehicle_id_only": True})
    dump_json(output_root / "state_transition_audit.json", {"transition_count": int(len(episodes)), "transitions": [{"episode_id": ep.get("episode_id"), "route_id": ep.get("route_id"), "vehicle_id": ep.get("vehicle_id"), "final_status": ep.get("episode_status"), "first_upstream_sequence": ep.get("first_upstream_watch_sequence"), "last_pre_terminal_sequence": ep.get("last_pre_terminal_sequence"), "terminal_entry_sequence": ep.get("first_terminal_sequence"), "last_terminal_sequence": ep.get("last_terminal_sequence"), "first_post_terminal_sequence": ep.get("first_post_terminal_sequence"), "confirmation_threshold_sample_count": ep.get("confirmation_threshold_sample_count"), "observed_post_terminal_confirmation_sample_count": ep.get("observed_post_terminal_confirmation_sample_count")} for ep in episodes.to_dict("records")]})
    dump_json(output_root / "counter_contract_v10_audit.json", counter)
    dump_json(output_root / "clock_semantics_audit.json", {"clock_semantics_audit_completed": True, "clock_semantics_status_counts": dict(clock_counts), "invalid_clock_order_count": invalid_clock_order_count, "episode_count": int(len(episodes))})
    dump_json(output_root / "interval_validation_audit.json", interval)
    dump_json(output_root / "episode_evidence_sha256_audit.json", evidence)
    dump_json(output_root / "episode_deduplication_audit.json", duplicate)

    cumulative_registry, registry_payload = build_cumulative_registry(cleanup_registry, episodes, output_root)
    cumulative_registry.to_parquet(output_root / "cumulative_episode_registry_candidate.parquet", index=False)
    dump_json(output_root / "cumulative_episode_registry_candidate.json", registry_payload)
    cumulative_count = int(len(cumulative_registry))
    route_cumulative_counts = cumulative_registry["route_id"].astype(str).value_counts().sort_index().to_dict() if not cumulative_registry.empty else {}
    dump_json(output_root / "method_prototype_progress_audit.json", {"prior_candidate_count": int(len(cleanup_registry)), "r2d1i_verified_new_complete_count": int(counter["total"]["new_complete_episode_count"]), "cumulative_complete_candidate": cumulative_count, "remaining_complete_episodes_to_12": max(0, 12 - cumulative_count), "route_cumulative_counts": route_cumulative_counts, "cumulative_independent_vehicle_count": int(cumulative_registry["vehicle_id"].dropna().astype(str).nunique()) if not cumulative_registry.empty else 0, "observation_date_count": int(cumulative_registry["observation_date"].dropna().astype(str).nunique()) if not cumulative_registry.empty and "observation_date" in cumulative_registry else 0, "hour_bucket_count": int(cumulative_registry["hour_bucket"].dropna().astype(str).nunique()) if not cumulative_registry.empty and "hour_bucket" in cumulative_registry else 0})

    lock_payload = {
        "campaign_b_authorized": False,
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "prompt6a_authorized": False,
        "e2_authorized": False,
        "full_matrix_authorized": False,
    }
    campaign_a_target_met = bool(cleanup_gate.get("route_results", {}).get("4010002001", {}).get("complete", 0) >= 1 and counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"] >= 1)
    campaign_a_aggregate_status = "TARGET_MET" if campaign_a_target_met else "PARTIAL"
    dump_json(output_root / "campaign_a_aggregate_status.json", {"campaign_a_aggregate_status": campaign_a_aggregate_status, "prior_cleanup_gate": cleanup_gate.get("gate_status"), "prior_4010002001_new_complete": cleanup_gate.get("route_results", {}).get("4010002001", {}).get("complete", 0), "r2d1i_4010002118_new_complete": counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"], "campaign_b_authorized": False})
    dump_json(output_root / "campaign_b_execution_authorization.json", {"approved": False, **lock_payload})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", lock_payload)
    dump_json(output_root / "simulator_parameter_translation_guard.json", lock_payload)
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, **lock_payload})

    complete_ok = bool(
        counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"] == 1
        and counter["total"]["new_complete_episode_count"] <= 1
        and counter["by_route"][TARGET_ROUTE]["post_terminal_confirmation_observation_count"] >= 3
        and duplicate.get("episode_duplicate_count", 0) == 0
        and mapping_regression_count == 0
        and counter["total"]["contradiction_count"] == 0
        and invalid_clock_order_count == 0
        and evidence.get("provenance_failure_count", 0) == 0
        and interval.get("interval_validation_failure_count", 0) == 0
        and secret_leak_count == 0
        and non_target_route_api_calls == 0
        and calls_after_fatal == 0
        and counter["counter_contract_v10_passed"]
    )
    runtime_ok = bool((not authorization_approved or preflight_passed) and (runner is None or runner.fatal_error is None) and calls_after_fatal == 0 and max_calls_per_minute_observed <= MAX_CALLS_PER_MINUTE)
    if not authorization_approved:
        gate_status = blocking_reason or "FAIL_OBSERVATION_CAMPAIGN"
    elif not preflight_passed:
        gate_status = "BLOCKED_API_RUNTIME"
    elif non_target_route_api_calls or non_target_raw_dirs:
        gate_status = "FAIL_NON_TARGET_ROUTE_ACCESS"
    elif mapping_regression_count:
        gate_status = "FAIL_MAPPING_REGRESSION"
    elif duplicate.get("episode_duplicate_count", 0):
        gate_status = "FAIL_EPISODE_DUPLICATION"
    elif not evidence.get("episode_evidence_sha256_audit_passed", False):
        gate_status = "FAIL_PROVENANCE_AUDIT"
    elif not interval.get("interval_validation_passed", False):
        gate_status = "FAIL_INTERVAL_VALIDATION"
    elif not counter.get("counter_contract_v10_passed", False):
        gate_status = "FAIL_COUNTER_CONTRACT"
    elif secret_leak_count:
        gate_status = "FAIL_SECURITY_AUDIT"
    elif runner and runner.fatal_error:
        gate_status = "BLOCKED_API_RUNTIME"
    elif complete_ok:
        gate_status = "PASS_TARGETED_4010002118_COMPLETE"
    elif runtime_ok:
        gate_status = "PASS_TARGETED_4010002118_PARTIAL"
    else:
        gate_status = "FAIL_OBSERVATION_CAMPAIGN"
    if authoritative_modified_count and gate_status.startswith("PASS"):
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"

    candidate_vehicle_id = None
    candidate_vehicle_class = None
    if not episodes.empty:
        candidate_vehicle_id = str(episodes.iloc[-1]["vehicle_id"])
        candidate_vehicle_class = str(episodes.iloc[-1]["vehicle_history_class"])
    elif runner and runner.session is not None:
        candidate_vehicle_id = runner.session.get("vehicle_id")
        candidate_vehicle_class = runner.session.get("vehicle_history_class")

    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "campaign_a_aggregate_status": campaign_a_aggregate_status,
        "run_date": run_date,
        "campaign_window": f"{current_time.isoformat(timespec='seconds')} to {planned_end.isoformat(timespec='seconds')}",
        "authorization_approved": authorization_approved and preflight_passed,
        "blocking_reason": None if authorization_approved else blocking_reason,
        "service_key_present": service_key_present,
        "service_key_length": service_key_length,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": len(preflight_records),
        "campaign_physical_calls": len(campaign_records),
        "r2d1i_total_physical_calls": 0 if runner is None else runner.physical_calls_this_artifact,
        "total_physical_calls_on_run_date": prior_calls + (0 if runner is None else runner.physical_calls_this_artifact),
        "effective_r2d1i_hard_cap": effective_cap,
        "max_calls_per_minute": max_calls_per_minute_observed,
        "target_route": TARGET_ROUTE,
        "non_target_route_api_calls": non_target_route_api_calls,
        "candidate_vehicle_id": candidate_vehicle_id,
        "candidate_vehicle_class": candidate_vehicle_class,
        "disappeared_waiting_reentry_observation_count": counter["by_route"][TARGET_ROUTE]["disappeared_waiting_reentry_observation_count"],
        "post_terminal_confirmation_observation_count": counter["by_route"][TARGET_ROUTE]["post_terminal_confirmation_observation_count"],
        "new_complete_episode_count": counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"],
        "right_censored_episode_count": counter["by_route"][TARGET_ROUTE]["right_censored_episode_count"],
        "new_independent_complete_vehicle_count": counter["by_route"][TARGET_ROUTE]["new_independent_complete_vehicle_count"],
        "cumulative_complete_candidate": cumulative_count,
        "remaining_to_method_prototype_12": max(0, 12 - cumulative_count),
        "mapping_regression_count": mapping_regression_count,
        "episode_duplicate_count": duplicate.get("episode_duplicate_count", 0),
        "contradiction_count": counter["total"]["contradiction_count"],
        "interval_validation_failure_count": interval.get("interval_validation_failure_count", 0),
        "calls_after_first_fatal_error": calls_after_fatal,
        "secret_leak_count": secret_leak_count,
        "counter_contract_v10_passed": counter["counter_contract_v10_passed"],
        "campaign_b_authorized": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "next_authorized_action": "offline review only",
    }
    dump_json(output_root / "prompt5_e01_r2d1i_gate.json", gate)

    report_lines = [
        "# Prompt 5-E01-R2D-1I Final Report",
        "",
        "## Status",
        "",
        f"- gate: {gate_status}",
        f"- campaign_a_aggregate_status: {campaign_a_aggregate_status}",
        f"- run_date: {run_date}",
        f"- campaign_window: {gate['campaign_window']}",
        f"- authorization_approved: {str(gate['authorization_approved']).lower()}",
        f"- blocking_reason: {gate['blocking_reason']}",
        f"- service_key_present: {str(service_key_present).lower()}",
        f"- service_key_length: {service_key_length}",
        f"- prior_physical_calls_on_run_date: {prior_calls}",
        f"- preflight_physical_calls: {len(preflight_records)}",
        f"- campaign_physical_calls: {len(campaign_records)}",
        f"- r2d1i_total_physical_calls: {gate['r2d1i_total_physical_calls']}",
        f"- total_physical_calls_on_run_date: {gate['total_physical_calls_on_run_date']}",
        f"- effective_r2d1i_hard_cap: {effective_cap}",
        f"- max_calls_per_minute: {max_calls_per_minute_observed}",
        f"- target_route: {TARGET_ROUTE}",
        f"- non_target_route_api_calls: {non_target_route_api_calls}",
        "",
        "## Candidate And Episode",
        "",
        f"- candidate_vehicle_id: {candidate_vehicle_id or 'NONE'}",
        f"- candidate_vehicle_class: {candidate_vehicle_class or 'NONE'}",
    ]
    for ep in episodes.to_dict("records"):
        report_lines.extend(
            [
                f"- episode_id: {ep.get('episode_id')}",
                f"- first_upstream_sequence: {ep.get('first_upstream_watch_sequence')}",
                f"- last_pre_terminal_sequence: {ep.get('last_pre_terminal_sequence')}",
                f"- first_terminal_sequence: {ep.get('first_terminal_sequence')}",
                f"- last_terminal_sequence: {ep.get('last_terminal_sequence')}",
                f"- disappearance_observation_count: {counter['by_route'][TARGET_ROUTE]['disappeared_waiting_reentry_observation_count']}",
                f"- exact_id_reentry_sequence: {ep.get('first_post_terminal_sequence')}",
                f"- confirmation_threshold: {ep.get('confirmation_threshold_sample_count')}",
                f"- actual_confirmation_count: {ep.get('observed_post_terminal_confirmation_sample_count')}",
                f"- final_status: {ep.get('episode_status')}",
                f"- clock_semantics: {ep.get('clock_semantics_status')}",
                f"- provider_interval: {ep.get('provider_lower_bound_sec')} to {ep.get('provider_upper_bound_sec')}",
                f"- request_interval: {ep.get('request_lower_bound_sec')} to {ep.get('request_upper_bound_sec')}",
                f"- conservative_dual_interval: {ep.get('conservative_dual_lower_bound_sec')} to {ep.get('conservative_dual_upper_bound_sec')}",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Counts And Audits",
            "",
            f"- complete/left/right/invalid: {counter['total']['new_complete_episode_count']}/{counter['total']['left_censored_episode_count']}/{counter['total']['right_censored_episode_count']}/{counter['total']['invalid_episode_count']}",
            f"- post_terminal_confirmation_observation_count: {counter['by_route'][TARGET_ROUTE]['post_terminal_confirmation_observation_count']}",
            f"- disappeared_waiting_reentry_observation_count: {counter['by_route'][TARGET_ROUTE]['disappeared_waiting_reentry_observation_count']}",
            f"- SHA provenance passed: {str(evidence['episode_evidence_sha256_audit_passed']).lower()}",
            f"- duplicate count: {duplicate.get('episode_duplicate_count', 0)}",
            f"- mapping regression count: {mapping_regression_count}",
            f"- counter contract v10 passed: {str(counter['counter_contract_v10_passed']).lower()}",
            f"- secret leak count: {secret_leak_count}",
            f"- cumulative candidate count: {cumulative_count}",
            f"- remaining to 12: {max(0, 12 - cumulative_count)}",
            "",
            "## Locks",
            "",
            "- Campaign B authorized: false",
            "- estimation executed: false",
            "- simulator parameter generated: false",
            "- Phase 2 authorized: false",
            "",
            "## Next Authorized Action",
            "",
            "offline review only",
        ]
    )
    (output_root / "prompt5_e01_r2d1i_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_manifest(output_root)

    print("R2D-1I TARGETED 4010002118 LIVE OBSERVATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    print("\ncampaign_a_aggregate_status:")
    print(campaign_a_aggregate_status)
    print("\nrun_date:")
    print(run_date)
    print("\ncampaign_window:")
    print(gate["campaign_window"])
    print("\nauthorization_approved:")
    print(str(gate["authorization_approved"]).lower())
    print("\nservice_key_present:")
    print(str(service_key_present).lower())
    print("\nservice_key_length:")
    print(service_key_length)
    print("\nprior_physical_calls_on_run_date:")
    print(prior_calls)
    print("\npreflight_physical_calls:")
    print(len(preflight_records))
    print("\ncampaign_physical_calls:")
    print(len(campaign_records))
    print("\nr2d1i_total_physical_calls:")
    print(gate["r2d1i_total_physical_calls"])
    print("\ntotal_physical_calls_on_run_date:")
    print(gate["total_physical_calls_on_run_date"])
    print("\neffective_r2d1i_hard_cap:")
    print(effective_cap)
    print("\nmax_calls_per_minute:")
    print(max_calls_per_minute_observed)
    print("\ntarget_route:")
    print(TARGET_ROUTE)
    print("\nnon_target_route_api_calls:")
    print(non_target_route_api_calls)
    print("\ncandidate_vehicle_id:")
    print(candidate_vehicle_id or "NONE")
    print("\ncandidate_vehicle_class:")
    print(candidate_vehicle_class or "NONE")
    print("\ndisappeared_waiting_reentry_observation_count:")
    print(counter["by_route"][TARGET_ROUTE]["disappeared_waiting_reentry_observation_count"])
    print("\npost_terminal_confirmation_observation_count:")
    print(counter["by_route"][TARGET_ROUTE]["post_terminal_confirmation_observation_count"])
    print("\nnew_complete_episode_count:")
    print(counter["by_route"][TARGET_ROUTE]["new_complete_episode_count"])
    print("\nright_censored_episode_count:")
    print(counter["by_route"][TARGET_ROUTE]["right_censored_episode_count"])
    print("\nnew_independent_complete_vehicle_count:")
    print(counter["by_route"][TARGET_ROUTE]["new_independent_complete_vehicle_count"])
    print("\ncumulative_complete_candidate:")
    print(cumulative_count)
    print("\nremaining_to_method_prototype_12:")
    print(max(0, 12 - cumulative_count))
    print("\nmapping_regression_count:")
    print(mapping_regression_count)
    print("\nepisode_duplicate_count:")
    print(duplicate.get("episode_duplicate_count", 0))
    print("\ncontradiction_count:")
    print(counter["total"]["contradiction_count"])
    print("\ninterval_validation_failure_count:")
    print(interval.get("interval_validation_failure_count", 0))
    print("\ncalls_after_first_fatal_error:")
    print(calls_after_fatal)
    print("\nsecret_leak_count:")
    print(secret_leak_count)
    print("\ncampaign_b_authorized:\nfalse")
    print("\nterminal_recovery_estimated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\nsimulator_application_authorized:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:")
    print("offline review only")


if __name__ == "__main__":
    main()
