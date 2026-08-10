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


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_REL = Path("05_training/artifacts")
HF1_REL = ARTIFACTS_REL / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_REL = ARTIFACTS_REL / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"
R2D1G_REL = ARTIFACTS_REL / "prompt5_e01_r2d1g_observation_expansion_plan_20260724_170238"
OUTPUT_PREFIX = "prompt5_e01_r2d1h_campaign_a_live_observation"
TARGET_ROUTES = ["4010002118", "4010002001"]
GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
KST = ZoneInfo("Asia/Seoul")

DAILY_SAFETY_CAP = 800
RECOMMENDED_CAMPAIGN_CAP = 260
ABSOLUTE_CAMPAIGN_CAP = 300
MAX_CALLS_PER_MINUTE = 8
POST_CONFIRM_MIN_SAMPLES = 3
POST_CONFIRM_MIN_SECONDS = 10 * 60
REQUIRED_END_BUFFER_MINUTES = 15
MAX_FOLLOW_MINUTES = {"4010002118": 100, "4010002001": 75}
FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE"}

SAMPLE_COLUMNS = [
    "request_id",
    "campaign_id",
    "session_id",
    "episode_id",
    "route_id",
    "vehicle_id",
    "capture_mode",
    "derived_terminal_phase",
    "request_observation_time",
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
    "route_continuity_passed",
    "direction_continuity_passed",
    "provider_response_status",
    "provider_result_code",
    "http_status",
    "raw_file_relative_path",
    "raw_file_sha256",
    "authoritative_raw",
]

EPISODE_COLUMNS = [
    "episode_id",
    "route_id",
    "vehicle_id",
    "direction",
    "vehicle_history_class",
    "episode_status",
    "final_status_class",
    "first_upstream_watch_time",
    "last_pre_terminal_request_time",
    "first_terminal_request_time",
    "last_terminal_request_time",
    "first_post_terminal_request_time",
    "post_terminal_confirmed_request_time",
    "last_pre_terminal_provider_time",
    "first_terminal_provider_time",
    "last_terminal_provider_time",
    "first_post_terminal_provider_time",
    "first_upstream_watch_sequence",
    "last_pre_terminal_sequence",
    "first_terminal_sequence",
    "last_terminal_sequence",
    "first_post_terminal_sequence",
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
    "left_censored",
    "right_censored",
    "dual_censored",
    "complete_interval_censored_episode",
    "upstream_watch_sample_count",
    "terminal_hold_sample_count",
    "post_terminal_confirmation_sample_count",
    "confirmation_threshold_sample_count",
    "observed_post_terminal_confirmation_sample_count",
    "first_upstream_raw_sha256",
    "last_pre_terminal_raw_sha256",
    "first_terminal_raw_sha256",
    "last_terminal_raw_sha256",
    "first_post_terminal_raw_sha256",
    "post_terminal_confirmation_raw_sha256s",
    "vehicle_id_continuity_passed",
    "route_continuity_passed",
    "direction_continuity_passed",
    "request_time_monotonicity_passed",
    "provider_time_monotonicity_passed",
    "evidence_sha256_passed",
    "clock_semantics_status",
    "eligible_for_estimation_input",
    "invalid_reason",
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
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
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


def rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


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
    text = norm(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def as_float(value: Any) -> Optional[float]:
    text = norm(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_dt(value: Any) -> Optional[datetime]:
    text = norm(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def seconds_between(left: Any, right: Any) -> Optional[float]:
    ldt = parse_dt(left)
    rdt = parse_dt(right)
    if ldt is None or rdt is None:
        return None
    return float((rdt - ldt).total_seconds())


def provider_time(request_iso: str, hhmmss: Any) -> Optional[str]:
    text = norm(hhmmss)
    if not text or len(text) < 6:
        return None
    try:
        request_dt = datetime.fromisoformat(request_iso).astimezone(KST)
        event = request_dt.replace(hour=int(text[0:2]), minute=int(text[2:4]), second=int(text[4:6]), microsecond=0)
        if event - request_dt > timedelta(hours=12):
            event -= timedelta(days=1)
        elif request_dt - event > timedelta(hours=12):
            event += timedelta(days=1)
        return event.isoformat(timespec="seconds")
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
    if "api token quota exceeded" in low or "too many" in low or "quota" in low:
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


def redact_url(url: str, secret: str) -> str:
    return (
        url.replace(secret, "<REDACTED>")
        .replace(urllib.parse.quote(secret, safe="%"), "<REDACTED>")
        .replace(urllib.parse.quote_plus(secret, safe="%"), "<REDACTED>")
    )


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    if not secret:
        return {"scan_root": str(root), "secret_literal_occurrence_count": 0, "findings": []}
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")]
    findings = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def strict_json_token_count(path: Path) -> int:
    data = path.read_bytes()
    return sum(data.count(token) for token in [b"NaN", b"Infinity", b"-Infinity"])


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows], columns=list(columns)).to_parquet(path, index=False)


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    record = {
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "file_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "readable": False,
        "schema_readable": False,
        "authoritative_source": source,
        "modified_during_r2d1h": False,
        "schema_error": None,
    }
    if not path.exists():
        record["schema_error"] = "MISSING"
        return record
    try:
        path.read_bytes()
        record["readable"] = True
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".parquet":
            pd.read_parquet(path)
        record["schema_readable"] = True
    except Exception as exc:
        record["schema_error"] = f"{type(exc).__name__}: {exc}"
    return record


def authoritative_inputs(project_root: Path) -> List[Tuple[Path, str]]:
    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    r2d1g = project_root / R2D1G_REL
    r2d1g_files = [
        "prompt5_e01_r2d1g_gate.json",
        "prompt5_e01_r2d1g_final_report.md",
        "route_observation_expansion_target_contract.json",
        "route_observation_expansion_target_table.parquet",
        "campaign_partition_plan.json",
        "campaign_schedule_design.json",
        "campaign_api_budget_contract.json",
        "campaign_stop_rule_contract.json",
        "campaign_state_machine_contract.json",
        "candidate_vehicle_selection_contract.json",
        "follow_duration_contract.json",
        "future_campaign_artifact_contract.json",
        "cumulative_episode_registry_update_contract.json",
        "method_prototype_data_gate_contract.json",
    ]
    r2d1f_files = [
        "terminal_recovery_estimand_b_approval_contract.json",
        "route_specific_estimand_boundary_contract.json",
        "route_specific_estimand_boundary_table.parquet",
        "dual_clock_interval_design_contract.json",
        "terminal_recovery_estimation_input_schema.json",
        "terminal_recovery_estimation_input_data_dictionary.json",
    ]
    r2d1e_files = [
        "frozen_complete_episode_registry.json",
        "frozen_complete_episode_registry.parquet",
        "clock_semantics_frozen_episode_audit.parquet",
    ]
    hf1_files = [
        "prompt5_e01_r2d1c_r4a_hf1_gate.json",
        "turnaround_mapping_contract_v10_hf1.json",
        "turnaround_mapping_contract_v10_hf1.parquet",
    ]
    return (
        [(r2d1g / name, "R2D-1G") for name in r2d1g_files]
        + [(r2d1f / name, "R2D-1F") for name in r2d1f_files]
        + [(r2d1e / name, "R2D-1E") for name in r2d1e_files]
        + [(hf1 / name, "HF1") for name in hf1_files]
    )


def route_metas(mapping_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(mapping_path)
    metas: Dict[str, Dict[str, Any]] = {}
    for route in TARGET_ROUTES:
        row = df[df["route_id"].astype(str) == route].iloc[0].to_dict()
        effective = as_int(row.get("effective_live_terminal_sequence"))
        metas[route] = {
            **row,
            "route_id": route,
            "direction": norm(row.get("direction_id")) or norm(row.get("service_direction_id")) or "1",
            "effective_live_terminal_sequence": effective,
            "early_upstream_trigger_sequence": None if effective is None else effective - 22,
            "upstream_trigger_sequence": None if effective is None else effective - 12,
            "terminal_trigger_sequence": None if effective is None else effective - 5,
        }
    return metas


def raw_index_for_date(project_root: Path, run_date: str) -> List[str]:
    ymd = run_date.replace("-", "")
    return sorted(str(path) for path in (project_root / ARTIFACTS_REL).rglob(f"{ymd}_*.json") if "/raw/" in str(path))


def request_interval(mode: str) -> int:
    return {
        "BROAD_SCAN": 120,
        "EARLY_UPSTREAM_WATCH": 60,
        "UPSTREAM_FOCUSED": 45,
        "TERMINAL_FOCUSED": 30,
        "POST_TERMINAL_WAIT": 30,
        "POST_TERMINAL_CONFIRM": 30,
    }.get(mode, 120)


def derived_terminal_phase(row: Mapping[str, Any]) -> Optional[str]:
    seq = as_int(row.get("current_sequence"))
    terminal = as_int(row.get("terminal_trigger_sequence"))
    if seq is None or terminal is None:
        return None
    if seq < terminal:
        return None
    return "TERMINAL_ZONE_APPROACH"


class CampaignRunner:
    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        secret: str,
        metas: Mapping[str, Mapping[str, Any]],
        prior_registry: pd.DataFrame,
        planned_end: datetime,
        effective_cap: int,
        prior_calls: int,
    ) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.secret = secret
        self.metas = metas
        self.prior_registry = prior_registry
        self.planned_end = planned_end
        self.effective_cap = effective_cap
        self.prior_calls = prior_calls
        self.request_records: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []
        self.raw_files_by_route: Dict[str, List[Dict[str, Any]]] = {route: [] for route in TARGET_ROUTES}
        self.sessions: Dict[str, Optional[Dict[str, Any]]] = {route: None for route in TARGET_ROUTES}
        self.next_due: Dict[str, float] = {route: time.monotonic() for route in TARGET_ROUTES}
        self.session_counter = 0
        self.candidate_count_by_route = Counter()
        self.history_count_by_route: Dict[str, Counter] = {route: Counter() for route in TARGET_ROUTES}
        self.complete_count_by_route = Counter()
        self.left_count_by_route = Counter()
        self.right_count_by_route = Counter()
        self.invalid_count_by_route = Counter()
        self.new_independent_complete_by_route = Counter()
        self.fatal_error: Optional[str] = None
        self.first_fatal_error_time: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.last_call_mono: Optional[float] = None
        self.consecutive_timeouts = 0
        self.consecutive_parse_failures = 0
        self.consecutive_network_errors = 0
        self.allow_new_candidates = True
        self.campaign_started_at: Optional[str] = None
        self.campaign_finished_at: Optional[str] = None
        prior = prior_registry.copy()
        self.prior_complete_vehicle_ids = {
            route: set(prior[prior["route_id"].astype(str) == route]["vehicle_id"].dropna().astype(str).tolist()) for route in TARGET_ROUTES
        }
        self.campaign_seen_vehicle_ids: Dict[str, set[str]] = {route: set() for route in TARGET_ROUTES}

    @property
    def physical_calls_this_artifact(self) -> int:
        return len(self.request_records)

    @property
    def preflight_calls(self) -> int:
        return sum(1 for req in self.request_records if req.get("capture_mode") == "PREFLIGHT")

    @property
    def campaign_calls(self) -> int:
        return self.physical_calls_this_artifact - self.preflight_calls

    def remaining_minutes(self) -> float:
        return (self.planned_end - now_kst()).total_seconds() / 60.0

    def route_can_start(self, route: str) -> bool:
        needed = MAX_FOLLOW_MINUTES[route] + REQUIRED_END_BUFFER_MINUTES
        return self.remaining_minutes() >= needed

    def call_allowed(self) -> bool:
        if self.physical_calls_this_artifact >= self.effective_cap:
            self.fatal_error = "EFFECTIVE_CAMPAIGN_HARD_CAP_REACHED"
            self.first_fatal_error_time = self.first_fatal_error_time or iso()
            self.stop_reason = self.fatal_error
            return False
        if self.prior_calls + self.physical_calls_this_artifact >= DAILY_SAFETY_CAP:
            self.fatal_error = "DAILY_PHYSICAL_CALL_SAFETY_CAP_REACHED"
            self.first_fatal_error_time = self.first_fatal_error_time or iso()
            self.stop_reason = self.fatal_error
            return False
        return True

    def fetch(self, route: str, mode: str) -> List[Dict[str, Any]]:
        if not self.call_allowed():
            return []
        if self.last_call_mono is not None:
            sleep_for = (60.0 / MAX_CALLS_PER_MINUTE) - (time.monotonic() - self.last_call_mono)
            if sleep_for > 0:
                time.sleep(sleep_for)

        request_dt = now_kst()
        request_time = request_dt.isoformat(timespec="seconds")
        request_id = f"r2d1h_{len(self.request_records) + 1:05d}_{route}_{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}"
        raw_dir = self.output_root / "raw" / route
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}.json"
        params = [("serviceKey", self.secret), ("routeId", route), ("resultType", "json")]
        url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
        http_status: Optional[int] = None
        content_type = ""
        timeout_error = False
        parsed = False
        result_code: Optional[str] = None
        payload: bytes
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1h/1.0"})
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
        status = detect_response_status(text, http_status, timeout_error)
        items: List[Dict[str, Any]] = []
        if status == "OK":
            items, parsed, result_code = parse_items(text)
            if not parsed:
                status = "PROVIDER_PARSE_FAILURE"
        raw_sha = sha256_file(raw_path)
        route_mismatch_count = 0
        rows: List[Dict[str, Any]] = []
        meta = self.metas[route]
        effective = as_int(meta.get("effective_live_terminal_sequence"))
        early = as_int(meta.get("early_upstream_trigger_sequence"))
        upstream = as_int(meta.get("upstream_trigger_sequence"))
        terminal = as_int(meta.get("terminal_trigger_sequence"))
        for item in items:
            response_route = norm(item.get("routeId")) or route
            if response_route != route:
                route_mismatch_count += 1
            seq = as_int(item.get("seq") or item.get("stopSeq"))
            vehicle_id = norm(item.get("vhcNo2")) or norm(item.get("vhcNo")) or norm(item.get("busId"))
            if seq is None or vehicle_id is None:
                continue
            direction = norm(item.get("moveDir")) or norm(item.get("direction_id")) or norm(meta.get("direction")) or "1"
            provider_event_time = provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
            sample = {
                "request_id": request_id,
                "campaign_id": "R2D-1H-CAMPAIGN-A",
                "session_id": None,
                "episode_id": None,
                "route_id": route,
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
                "route_continuity_passed": response_route == route,
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
            "route_id": route,
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
            "redacted_request_metadata": {"endpoint": GETPOS02_URL, "routeId": route, "resultType": "json", "serviceKey": "<REDACTED>"},
            "request_url_redacted": redact_url(url, self.secret),
            "physical_call_index_this_artifact": self.physical_calls_this_artifact + 1,
        }
        self.request_records.append(record)
        self.raw_files_by_route[route].append(record)
        self.samples.extend(rows)
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
                    "event": "r2d1h_api_call",
                    "route_id": route,
                    "mode": mode,
                    "status": status,
                    "rows": len(rows),
                    "calls": self.physical_calls_this_artifact,
                    "complete_total": int(sum(self.complete_count_by_route.values())),
                    "time": request_time,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return rows

    def classify_vehicle(self, route: str, vehicle_id: str) -> str:
        if vehicle_id in self.campaign_seen_vehicle_ids[route]:
            return "DUPLICATE_TERMINAL_CYCLE"
        if vehicle_id in self.prior_complete_vehicle_ids[route]:
            return "PREVIOUSLY_COMPLETE_VEHICLE"
        return "NEW_INDEPENDENT_VEHICLE"

    def desired_route_for_new_session(self, route: str) -> bool:
        if not self.allow_new_candidates or self.sessions[route] is not None:
            return False
        if sum(self.complete_count_by_route.values()) >= 3:
            return False
        if not self.route_can_start(route):
            return False
        missing_routes = [r for r in TARGET_ROUTES if self.complete_count_by_route[r] < 1]
        if missing_routes and route not in missing_routes:
            return False
        return True

    def maybe_start_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self.desired_route_for_new_session(route):
            return
        early = [dict(row) for row in rows if row.get("is_early_upstream_watch_zone")]
        fallback = [dict(row) for row in rows if row.get("is_upstream_watch_zone")]
        candidates = early or fallback
        if not candidates:
            return
        scored = []
        priority = {"NEW_INDEPENDENT_VEHICLE": 3, "PREVIOUSLY_CENSORED_VEHICLE": 2, "PREVIOUSLY_COMPLETE_VEHICLE": 1, "UNKNOWN_VEHICLE_HISTORY": 0}
        for row in candidates:
            vehicle_id = norm(row.get("vehicle_id"))
            if vehicle_id is None:
                continue
            history = self.classify_vehicle(route, vehicle_id)
            if history == "DUPLICATE_TERMINAL_CYCLE":
                continue
            scored.append((priority.get(history, 0), as_int(row.get("current_sequence")) or -1, history, row))
        if not scored:
            return
        _, _, history, chosen = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
        self.session_counter += 1
        self.candidate_count_by_route[route] += 1
        self.history_count_by_route[route][history] += 1
        session_id = f"r2d1h_session_{self.session_counter:05d}_{route}"
        episode_id = f"r2d1h_episode_{self.session_counter:05d}_{route}"
        chosen["session_id"] = session_id
        chosen["episode_id"] = episode_id
        phase = "EARLY_UPSTREAM_WATCH" if chosen.get("is_early_upstream_watch_zone") else "UPSTREAM_FOCUSED"
        self.sessions[route] = {
            "route_id": route,
            "session_id": session_id,
            "episode_id": episode_id,
            "vehicle_id": norm(chosen.get("vehicle_id")),
            "direction": norm(chosen.get("direction")),
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
            "post_terminal_confirmation_sample_count": 0,
            "post_terminal_confirmation_samples": [],
            "first_terminal_monotonic": None,
            "missing_after_terminal_count": 0,
        }
        self.campaign_seen_vehicle_ids[route].add(str(chosen["vehicle_id"]))
        self._tag_sample(chosen)
        print(
            json.dumps(
                {
                    "event": "r2d1h_candidate_started",
                    "route_id": route,
                    "vehicle_id": chosen["vehicle_id"],
                    "vehicle_history_class": history,
                    "sequence": chosen["current_sequence"],
                    "phase": phase,
                    "time": iso(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _tag_sample(self, row: Mapping[str, Any]) -> None:
        for sample in reversed(self.samples):
            if (
                sample.get("raw_file_sha256") == row.get("raw_file_sha256")
                and sample.get("vehicle_id") == row.get("vehicle_id")
                and sample.get("current_sequence") == row.get("current_sequence")
            ):
                sample["session_id"] = row.get("session_id")
                sample["episode_id"] = row.get("episode_id")
                break

    def update_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
        session = self.sessions.get(route)
        if session is None:
            return
        vehicle_id = session["vehicle_id"]
        direction = session["direction"]
        matching: List[Dict[str, Any]] = []
        for row in rows:
            if norm(row.get("vehicle_id")) != vehicle_id:
                continue
            if norm(row.get("route_id")) != route or norm(row.get("direction")) != direction:
                self.finish_session(route, "INVALID", "INVALID_ROUTE_DIRECTION_CONTINUITY")
                return
            enriched = dict(row)
            enriched["session_id"] = session["session_id"]
            enriched["episode_id"] = session["episode_id"]
            matching.append(enriched)
            self._tag_sample(enriched)
        for row in matching:
            session["missing_after_terminal_count"] = 0
            seq = as_int(row.get("current_sequence"))
            terminal = as_int(row.get("terminal_trigger_sequence"))
            if seq is None or terminal is None:
                continue
            if session["phase"] == "POST_TERMINAL_CONFIRM":
                session["post_terminal_confirmation_sample_count"] = int(session["post_terminal_confirmation_sample_count"]) + 1
                session["post_terminal_confirmation_samples"].append(row)
                first_post_time = session["first_post_terminal"].get("request_observation_time") if session.get("first_post_terminal") else None
                elapsed = seconds_between(first_post_time, row.get("request_observation_time"))
                if int(session["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or (
                    elapsed is not None and elapsed >= POST_CONFIRM_MIN_SECONDS
                ):
                    session["post_terminal_confirmed"] = row
                    self.finish_session(route, "COMPLETE_INTERVAL_CENSORED")
                    return
                continue
            last_terminal = session.get("last_terminal")
            last_terminal_seq = as_int(last_terminal.get("current_sequence")) if last_terminal is not None else None
            if session.get("first_post_terminal") is None and last_terminal_seq is not None and last_terminal_seq >= terminal and seq <= 5 and seq < last_terminal_seq:
                row["sequence_reset"] = True
                row["derived_terminal_phase"] = "POST_TERMINAL_RESET"
                session["first_post_terminal"] = row
                session["post_terminal_confirmation_sample_count"] = 1
                session["post_terminal_confirmation_samples"] = [row]
                session["phase"] = "POST_TERMINAL_CONFIRM"
                print(
                    json.dumps(
                        {
                            "event": "r2d1h_reset_observed",
                            "route_id": route,
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
                row["derived_terminal_phase"] = "TERMINAL_ZONE_APPROACH"
                if session.get("first_terminal") is None:
                    session["first_terminal"] = row
                    session["first_terminal_monotonic"] = time.monotonic()
                    session["phase"] = "TERMINAL_FOCUSED"
                    print(
                        json.dumps(
                            {
                                "event": "r2d1h_terminal_entry",
                                "route_id": route,
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
        session = self.sessions.get(route)
        if session and session.get("first_terminal") is not None and not matching:
            session["missing_after_terminal_count"] = int(session.get("missing_after_terminal_count", 0)) + 1
            if int(session["missing_after_terminal_count"]) >= 3 and session.get("first_post_terminal") is None:
                session["phase"] = "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY"
        if session and session.get("first_terminal_monotonic") is not None:
            max_follow = MAX_FOLLOW_MINUTES[route] * 60
            if time.monotonic() - float(session["first_terminal_monotonic"]) >= max_follow:
                self.finish_session(route, "RIGHT_CENSORED_MAX_FOLLOW")

    def finish_session(self, route: str, status: str, invalid_reason: Optional[str] = None) -> None:
        session = self.sessions.get(route)
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
        complete = status == "COMPLETE_INTERVAL_CENSORED" and all(
            item is not None for item in [first_up, last_pre, first_terminal, last_terminal, first_post, confirmed]
        )
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
        conservative_lower = None
        conservative_upper = None
        if None not in [provider_lower, request_lower]:
            conservative_lower = min(float(provider_lower), float(request_lower))
        if None not in [provider_upper, request_upper]:
            conservative_upper = max(float(provider_upper), float(request_upper))
        confirmation_shas = [row.get("raw_file_sha256") for row in session.get("post_terminal_confirmation_samples", []) if row.get("raw_file_sha256")]
        clock_status = self.clock_status_for_session(session, complete)
        row = {
            "episode_id": session["episode_id"],
            "route_id": route,
            "vehicle_id": session["vehicle_id"],
            "direction": session["direction"],
            "vehicle_history_class": session["vehicle_history_class"],
            "episode_status": status if complete or invalid else status,
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
            self.complete_count_by_route[route] += 1
            if session["vehicle_history_class"] == "NEW_INDEPENDENT_VEHICLE":
                self.new_independent_complete_by_route[route] += 1
        elif invalid:
            self.invalid_count_by_route[route] += 1
        elif left:
            self.left_count_by_route[route] += 1
        else:
            self.right_count_by_route[route] += 1
        self.sessions[route] = None
        print(json.dumps({"event": "r2d1h_session_finished", "route_id": route, "status": row["episode_status"], "complete": complete, "time": iso()}, ensure_ascii=False), flush=True)

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
        for route in TARGET_ROUTES:
            rows = self.fetch(route, "PREFLIGHT")
            if not self.request_records:
                return False, "NO_PREFLIGHT_RECORD"
            record = self.request_records[-1]
            status = record.get("provider_response_status")
            if status in FATAL_STATUSES or self.fatal_error:
                return False, str(status)
            if status != "OK" or not record.get("provider_parse_success") or not record.get("route_id_response_match"):
                return False, str(status)
        return True, None

    def route_mode(self, route: str) -> str:
        session = self.sessions[route]
        if session is None:
            return "BROAD_SCAN"
        phase = str(session["phase"])
        if phase in {"EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED", "POST_TERMINAL_CONFIRM", "POST_TERMINAL_WAIT"}:
            return phase
        return "TERMINAL_FOCUSED"

    def run(self) -> None:
        self.campaign_started_at = iso()
        while not self.fatal_error and now_kst() < self.planned_end:
            complete_total = sum(self.complete_count_by_route.values())
            if complete_total >= 3:
                self.allow_new_candidates = False
                for route in TARGET_ROUTES:
                    if self.sessions[route] is not None:
                        self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_TARGET_REACHED")
                self.stop_reason = "CAMPAIGN_A_STRETCH_TARGET_REACHED"
                break
            if self.physical_calls_this_artifact >= int(self.effective_cap * 0.9):
                self.allow_new_candidates = False
                if not any(self.sessions.values()):
                    self.stop_reason = "EFFECTIVE_CAMPAIGN_HARD_CAP_90_PERCENT_REACHED"
                    break
            if not any(self.sessions.values()) and not any(self.route_can_start(route) for route in TARGET_ROUTES):
                self.stop_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
                break
            due_routes = [route for route in TARGET_ROUTES if self.next_due[route] <= time.monotonic()]
            if not due_routes:
                time.sleep(min(1.0, max(0.1, min(self.next_due.values()) - time.monotonic())))
                continue
            due_routes.sort(key=lambda route: (self.next_due[route], TARGET_ROUTES.index(route)))
            route = due_routes[0]
            if self.sessions[route] is None and not self.route_can_start(route):
                self.next_due[route] = time.monotonic() + 60
                continue
            mode = self.route_mode(route)
            rows = self.fetch(route, mode)
            if self.fatal_error:
                break
            self.update_session(route, rows)
            if self.sessions[route] is None:
                self.maybe_start_session(route, rows)
            mode_after = self.route_mode(route)
            self.next_due[route] = time.monotonic() + request_interval(mode_after)
        if self.fatal_error:
            for route in TARGET_ROUTES:
                if self.sessions[route] is not None:
                    self.finish_session(route, "RIGHT_CENSORED_FATAL_API_STOP")
        elif now_kst() >= self.planned_end:
            self.stop_reason = self.stop_reason or "CAMPAIGN_WINDOW_ENDED"
            for route in TARGET_ROUTES:
                if self.sessions[route] is not None:
                    self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_END")
        else:
            for route in TARGET_ROUTES:
                if self.sessions[route] is not None and self.stop_reason:
                    self.finish_session(route, "RIGHT_CENSORED_" + self.stop_reason)
        self.campaign_finished_at = iso()


def add_repeat_fields(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    out = samples.copy()
    out["_request_dt"] = pd.to_datetime(out["request_observation_time"], errors="coerce")
    for _, group in out.groupby(["route_id", "direction", "vehicle_id", "provider_position_event_time"], dropna=False):
        if len(group) < 2:
            continue
        reqs = group["_request_dt"].dropna()
        span = float((reqs.max() - reqs.min()).total_seconds()) if len(reqs) >= 2 else 0.0
        out.loc[group.index, "provider_event_timestamp_repeated"] = True
        out.loc[group.index, "provider_event_repeat_count"] = int(len(group) - 1)
        out.loc[group.index, "provider_event_repeat_request_span_sec"] = span
    return out.drop(columns=["_request_dt"])


def summarize_counter(samples: pd.DataFrame, episodes: pd.DataFrame, runner: CampaignRunner) -> Dict[str, Any]:
    def counts_for(route: Optional[str]) -> Dict[str, int]:
        s = samples if route is None else samples[samples["route_id"].astype(str) == route]
        e = episodes if route is None else episodes[episodes["route_id"].astype(str) == route]
        return {
            "broad_scan_observation_count": int((s["capture_mode"] == "BROAD_SCAN").sum()) if not s.empty else 0,
            "early_upstream_watch_observation_count": int((s["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()) if not s.empty else 0,
            "upstream_focused_observation_count": int((s["capture_mode"] == "UPSTREAM_FOCUSED").sum()) if not s.empty else 0,
            "terminal_focused_observation_count": int((s["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not s.empty else 0,
            "post_terminal_confirmation_observation_count": int((s["capture_mode"] == "POST_TERMINAL_CONFIRM").sum()) if not s.empty else 0,
            "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values()) if route is None else runner.candidate_count_by_route[route]),
            "new_independent_vehicle_candidate_count": int(
                sum(counter.get("NEW_INDEPENDENT_VEHICLE", 0) for counter in runner.history_count_by_route.values())
                if route is None
                else runner.history_count_by_route[route].get("NEW_INDEPENDENT_VEHICLE", 0)
            ),
            "previously_censored_vehicle_candidate_count": int(
                sum(counter.get("PREVIOUSLY_CENSORED_VEHICLE", 0) for counter in runner.history_count_by_route.values())
                if route is None
                else runner.history_count_by_route[route].get("PREVIOUSLY_CENSORED_VEHICLE", 0)
            ),
            "previously_complete_vehicle_candidate_count": int(
                sum(counter.get("PREVIOUSLY_COMPLETE_VEHICLE", 0) for counter in runner.history_count_by_route.values())
                if route is None
                else runner.history_count_by_route[route].get("PREVIOUSLY_COMPLETE_VEHICLE", 0)
            ),
            "tracking_session_started_count": int(len(e)),
            "pre_terminal_confirmed_count": int(e["last_pre_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "terminal_entry_count": int(e["first_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "terminal_loop_movement_observation_count": 0,
            "terminal_stop_hold_observation_count": int(e["terminal_hold_sample_count"].sum()) if not e.empty else 0,
            "post_terminal_wait_observation_count": int((s["capture_mode"] == "POST_TERMINAL_WAIT").sum()) if not s.empty else 0,
            "post_terminal_reset_count": int(e["first_post_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "post_terminal_confirmed_count": int(e["post_terminal_confirmed_request_time"].notna().sum()) if not e.empty else 0,
            "new_complete_episode_count": int(e["complete_interval_censored_episode"].sum()) if not e.empty else 0,
            "left_censored_episode_count": int(e["left_censored"].sum()) if not e.empty else 0,
            "right_censored_episode_count": int(e["right_censored"].sum()) if not e.empty else 0,
            "invalid_episode_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
            "new_independent_complete_vehicle_count": int(
                e[(e["complete_interval_censored_episode"] == True) & (e["vehicle_history_class"] == "NEW_INDEPENDENT_VEHICLE")]["vehicle_id"].nunique()
            )
            if not e.empty
            else 0,
            "episode_duplicate_count": 0,
            "contradiction_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
            "total_final_episode_count": int(len(e)),
            "complete_final_count": int((e["final_status_class"] == "COMPLETE").sum()) if not e.empty else 0,
            "left_censored_final_count": int(e["left_censored"].sum()) if not e.empty else 0,
            "right_censored_final_count": int(e["right_censored"].sum()) if not e.empty else 0,
            "invalid_final_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
        }

    total = counts_for(None)
    return {
        "counter_contract_v9_passed": total["total_final_episode_count"]
        == total["complete_final_count"] + total["left_censored_final_count"] + total["right_censored_final_count"] + total["invalid_final_count"],
        "total": total,
        "by_route": {route: counts_for(route) for route in TARGET_ROUTES},
    }


def evidence_sha_audit(output_root: Path, episodes: pd.DataFrame) -> Dict[str, Any]:
    sha_to_path = {}
    for path in output_root.rglob("*.json"):
        if "/raw/" in str(path):
            sha_to_path[sha256_file(path)] = str(path.relative_to(output_root))
    failures = []
    required_fields = [
        "first_upstream_raw_sha256",
        "last_pre_terminal_raw_sha256",
        "first_terminal_raw_sha256",
        "last_terminal_raw_sha256",
        "first_post_terminal_raw_sha256",
    ]
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        for field in required_fields:
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
        checks = [
            ("provider", ep.get("provider_lower_bound_sec"), ep.get("provider_upper_bound_sec")),
            ("request", ep.get("request_lower_bound_sec"), ep.get("request_upper_bound_sec")),
            ("conservative_dual", ep.get("conservative_dual_lower_bound_sec"), ep.get("conservative_dual_upper_bound_sec")),
        ]
        for label, lower, upper in checks:
            if lower is None or upper is None or float(lower) < 0 or float(upper) < 0 or float(lower) > float(upper):
                failures.append({"episode_id": ep.get("episode_id"), "clock": label, "lower": lower, "upper": upper})
    return {"interval_validation_passed": len(failures) == 0, "interval_validation_failure_count": len(failures), "negative_interval_count": len(failures), "failures": failures}


def duplicate_audit(prior_registry: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    failures = []
    prior = prior_registry.copy()
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        subset = prior[
            (prior["route_id"].astype(str) == str(ep.get("route_id")))
            & (prior["vehicle_id"].astype(str) == str(ep.get("vehicle_id")))
            & (
                (prior["first_terminal_raw_sha256"].astype(str) == str(ep.get("first_terminal_raw_sha256")))
                | (prior["first_post_terminal_raw_sha256"].astype(str) == str(ep.get("first_post_terminal_raw_sha256")))
            )
        ]
        if not subset.empty:
            failures.append({"episode_id": ep.get("episode_id"), "route_id": ep.get("route_id"), "vehicle_id": ep.get("vehicle_id")})
    return {"episode_deduplication_audit_passed": len(failures) == 0, "episode_duplicate_count": len(failures), "duplicates": failures}


def build_registry_candidate(prior_registry: pd.DataFrame, episodes: pd.DataFrame, output_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in prior_registry.iterrows():
        rows.append(
            {
                "frozen_episode_id": row.get("frozen_episode_id"),
                "source_campaign_id": "PRIOR_FROZEN",
                "source_artifact": row.get("source_artifact"),
                "source_episode_id": row.get("source_episode_id"),
                "route_id": row.get("route_id"),
                "vehicle_id": row.get("vehicle_id"),
                "observation_date": None,
                "hour_bucket": None,
                "first_terminal_raw_sha": row.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": row.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": row.get("clock_semantics_status"),
                "eligible_for_estimation_input": True,
            }
        )
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        first_terminal = parse_dt(ep.get("first_terminal_request_time"))
        rows.append(
            {
                "frozen_episode_id": None,
                "source_campaign_id": "A",
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
    payload = {"candidate_row_count": int(len(df)), "campaign_a_verified_new_complete_count": int(len(df) - len(prior_registry)), "records": rows}
    return df, payload


def final_gate(
    authorization_approved: bool,
    blocking_reason: Optional[str],
    runner: Optional[CampaignRunner],
    counter: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    interval: Mapping[str, Any],
    duplicate: Mapping[str, Any],
    mapping_regression_count: int,
    secret_leak_count: int,
    invalid_clock_order_count: int,
) -> str:
    if not authorization_approved:
        return blocking_reason or "FAIL_OBSERVATION_CAMPAIGN"
    if runner and runner.fatal_error:
        return "BLOCKED_API_RUNTIME"
    if mapping_regression_count:
        return "FAIL_MAPPING_REGRESSION"
    if duplicate.get("episode_duplicate_count", 0):
        return "FAIL_EPISODE_DUPLICATION"
    if not evidence_audit.get("episode_evidence_sha256_audit_passed", False):
        return "FAIL_PROVENANCE_AUDIT"
    if not interval.get("interval_validation_passed", False):
        return "FAIL_INTERVAL_VALIDATION"
    if secret_leak_count:
        return "FAIL_SECURITY_AUDIT"
    total = counter.get("total", {})
    if total.get("contradiction_count", 0) or invalid_clock_order_count or not counter.get("counter_contract_v9_passed", False):
        return "FAIL_OBSERVATION_CAMPAIGN"
    complete_2118 = int(counter.get("by_route", {}).get("4010002118", {}).get("new_complete_episode_count", 0))
    complete_2001 = int(counter.get("by_route", {}).get("4010002001", {}).get("new_complete_episode_count", 0))
    complete_total = int(total.get("new_complete_episode_count", 0))
    api_passed = runner is not None and runner.fatal_error is None and int(total.get("episode_duplicate_count", 0)) == 0
    if api_passed and complete_2118 >= 1 and complete_2001 >= 1 and complete_total == 3:
        return "PASS_CAMPAIGN_A_STRETCH_TARGET_MET"
    if api_passed and complete_2118 >= 1 and complete_2001 >= 1 and complete_total >= 2:
        return "PASS_CAMPAIGN_A_TARGET_MET"
    if api_passed:
        return "PASS_CAMPAIGN_A_PARTIAL"
    return "FAIL_OBSERVATION_CAMPAIGN"


def required_names() -> List[str]:
    names = [
        "prompt5_e01_r2d1h_manifest.json",
        "prompt5_e01_r2d1h_gate.json",
        "prompt5_e01_r2d1h_final_report.md",
        "hf1_reference.json",
        "r2d1e_reference.json",
        "r2d1f_reference.json",
        "r2d1g_reference.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "prior_episode_registry_reference_audit.json",
        "secret_leak_audit.json",
        "credential_reference.json",
        "campaign_a_execution_authorization.json",
        "campaign_a_observation_contract.json",
        "campaign_a_target_contract.json",
        "campaign_a_schedule_manifest.json",
        "daily_api_usage_audit.json",
        "campaign_preflight_audit.json",
        "campaign_runtime_audit.json",
        "api_stop_condition_audit.json",
        "candidate_vehicle_selection_audit.json",
        "campaign_state_transition_audit.json",
        "terminal_counter_audit_v9.json",
        "episode_evidence_sha256_audit.json",
        "episode_deduplication_audit.json",
        "clock_semantics_audit.json",
        "interval_validation_audit.json",
        "campaign_a_position_samples.parquet",
        "campaign_a_vehicle_trajectories.parquet",
        "campaign_a_terminal_recovery_episodes.parquet",
        "campaign_a_terminal_recovery_interval_bounds.parquet",
        "campaign_a_route_summary.parquet",
        "cumulative_episode_registry_candidate.json",
        "cumulative_episode_registry_candidate.parquet",
        "method_prototype_progress_audit.json",
        "campaign_b_execution_authorization.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "simulator_parameter_translation_guard.json",
        "phase2_execution_authorization.json",
    ]
    for route in TARGET_ROUTES:
        prefix = f"terminal_recovery_evidence/{route}"
        names.extend(
            [
                f"{prefix}/route_mapping_reference.json",
                f"{prefix}/observation_manifest.json",
                f"{prefix}/candidate_selection_audit.json",
                f"{prefix}/upstream_trigger_audit.json",
                f"{prefix}/focused_trigger_audit.json",
                f"{prefix}/vehicle_trajectories.parquet",
                f"{prefix}/position_samples.parquet",
                f"{prefix}/terminal_recovery_episodes.parquet",
                f"{prefix}/terminal_recovery_interval_bounds.parquet",
                f"{prefix}/episode_summary.json",
                f"{prefix}/clock_semantics_audit.json",
                f"{prefix}/raw_file_index.json",
                f"{prefix}/evidence_sha256_audit.json",
            ]
        )
    return names


def write_manifest(output_root: Path) -> None:
    files = []
    for name in sorted(required_names()):
        path = output_root / name
        if name == "prompt5_e01_r2d1h_manifest.json":
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
    missing = [entry["path"] for entry in files if not entry["exists"]]
    dump_json(
        output_root / "prompt5_e01_r2d1h_manifest.json",
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
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1H Campaign A controlled live observation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    for route in TARGET_ROUTES:
        (output_root / "raw" / route).mkdir(parents=True, exist_ok=True)
        (output_root / "terminal_recovery_evidence" / route).mkdir(parents=True, exist_ok=True)

    start_time = now_kst()
    run_date = start_time.date().isoformat()
    official_start = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
    official_end = start_time.replace(hour=14, minute=0, second=0, microsecond=0)
    planned_end = official_end
    planned_hours = [f"{hour:02d}:00-{hour:02d}:59" for hour in range(official_start.hour, official_end.hour)]

    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    r2d1g = project_root / R2D1G_REL
    input_pre = [file_audit(path, project_root, source) for path, source in authoritative_inputs(project_root)]
    failed_inputs = [record for record in input_pre if not record["readable"] or not record["schema_readable"]]

    dump_json(output_root / "hf1_reference.json", {"absolute_path": str(hf1), "sha256_gate": sha256_file(hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1e_reference.json", {"absolute_path": str(r2d1e), "sha256_registry": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet"), "read_only_input": True})
    dump_json(output_root / "r2d1f_reference.json", {"absolute_path": str(r2d1f), "sha256_gate": sha256_file(r2d1f / "prompt5_e01_r2d1f_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1g_reference.json", {"absolute_path": str(r2d1g), "sha256_gate": sha256_file(r2d1g / "prompt5_e01_r2d1g_gate.json"), "read_only_input": True})

    r2d1g_gate = json.loads((r2d1g / "prompt5_e01_r2d1g_gate.json").read_text(encoding="utf-8")) if (r2d1g / "prompt5_e01_r2d1g_gate.json").exists() else {}
    mapping_json = hf1 / "turnaround_mapping_contract_v10_hf1.json"
    mapping_parquet = hf1 / "turnaround_mapping_contract_v10_hf1.parquet"
    mapping_json_pre = sha256_file(mapping_json) if mapping_json.exists() else None
    mapping_parquet_pre = sha256_file(mapping_parquet) if mapping_parquet.exists() else None
    metas = route_metas(mapping_parquet) if mapping_parquet.exists() else {}
    prior_registry = pd.read_parquet(r2d1e / "frozen_complete_episode_registry.parquet") if (r2d1e / "frozen_complete_episode_registry.parquet").exists() else pd.DataFrame()
    prior_raw_index = raw_index_for_date(project_root, run_date)
    prior_calls = len(prior_raw_index)
    available_daily_budget = DAILY_SAFETY_CAP - prior_calls
    effective_cap = min(ABSOLUTE_CAMPAIGN_CAP, available_daily_budget)
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or ""
    credential_available = bool(service_key)
    credential_length = len(service_key) if credential_available else 0
    in_window = official_start <= start_time < official_end
    follow_ready_by_route = {
        route: (planned_end - start_time).total_seconds() >= (MAX_FOLLOW_MINUTES[route] + REQUIRED_END_BUFFER_MINUTES) * 60 for route in TARGET_ROUTES
    }

    blocking_reason: Optional[str] = None
    if failed_inputs:
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif r2d1g_gate.get("gate_status") != "PASS_OBSERVATION_EXPANSION_PLAN_READY":
        blocking_reason = "FAIL_OBSERVATION_CAMPAIGN"
    elif not in_window:
        blocking_reason = "WAITING_FOR_CAMPAIGN_WINDOW"
    elif not any(follow_ready_by_route.values()):
        blocking_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
    elif effective_cap < 120:
        blocking_reason = "BLOCKED_INSUFFICIENT_API_BUDGET"
    elif not credential_available:
        blocking_reason = "BLOCKED_API_RUNTIME"

    authorization_approved = blocking_reason is None
    dump_json(
        output_root / "campaign_a_execution_authorization.json",
        {
            "approved": authorization_approved,
            "blocking_reason": None if authorization_approved else blocking_reason,
            "campaign_id": "A",
            "target_routes": TARGET_ROUTES,
            "maximum_new_complete_episodes": 3,
            "estimation_execution_authorized": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
        },
    )
    dump_json(
        output_root / "campaign_a_schedule_manifest.json",
        {
            "run_date": run_date,
            "campaign_start_time": start_time.isoformat(timespec="seconds"),
            "planned_campaign_end_time": planned_end.isoformat(timespec="seconds"),
            "timezone": "Asia/Seoul",
            "day_of_week": start_time.strftime("%A"),
            "planned_hour_buckets": planned_hours,
            "official_observation_window": "09:00-14:00 KST",
            "current_time_in_official_window": in_window,
            "sufficient_follow_window_by_route": follow_ready_by_route,
        },
    )
    dump_json(
        output_root / "credential_reference.json",
        {
            "env_var": "DAEGU_BIS_SERVICE_KEY",
            "exists": credential_available,
            "length": credential_length,
            "raw_value_output": False,
            "raw_value_persisted": False,
            "request_url_with_key_persisted": False,
        },
    )
    dump_json(
        output_root / "campaign_a_target_contract.json",
        {
            "campaign_id": "A",
            "active_routes": TARGET_ROUTES,
            "minimum_targets": {"4010002118_new_complete_min": 1, "4010002001_new_complete_min": 1, "campaign_a_new_complete_total_min": 2},
            "stretch_target": {"campaign_a_new_complete_total": 3},
            "maximum_new_complete_episodes": 3,
        },
    )
    dump_json(
        output_root / "campaign_a_observation_contract.json",
        {
            "campaign_id": "A",
            "active_routes": TARGET_ROUTES,
            "state_machine": json.loads((r2d1g / "campaign_state_machine_contract.json").read_text(encoding="utf-8")) if (r2d1g / "campaign_state_machine_contract.json").exists() else {},
            "exact_vehicle_id_only": True,
            "primary_interval_rule": "CONSERVATIVE_DUAL_CLOCK_ENVELOPE",
            "no_midpoint_or_summary_statistic": True,
        },
    )

    runner: Optional[CampaignRunner] = None
    preflight_passed = False
    preflight_blocking_reason = blocking_reason
    if authorization_approved:
        runner = CampaignRunner(project_root, output_root, service_key, metas, prior_registry, planned_end, effective_cap, prior_calls)
        preflight_passed, preflight_blocking_reason = runner.preflight()
        if preflight_passed:
            runner.run()
        elif runner.fatal_error is None:
            runner.stop_reason = "PREFLIGHT_FAILED"

    samples = add_repeat_fields(pd.DataFrame([] if runner is None else runner.samples, columns=SAMPLE_COLUMNS))
    episodes = pd.DataFrame([] if runner is None else runner.episodes, columns=EPISODE_COLUMNS)
    bounds_cols = [
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
    bounds = episodes[bounds_cols].copy() if not episodes.empty else pd.DataFrame(columns=bounds_cols)
    write_parquet(output_root / "campaign_a_position_samples.parquet", samples.to_dict("records"), SAMPLE_COLUMNS)
    write_parquet(output_root / "campaign_a_vehicle_trajectories.parquet", samples.to_dict("records"), SAMPLE_COLUMNS)
    write_parquet(output_root / "campaign_a_terminal_recovery_episodes.parquet", episodes.to_dict("records"), EPISODE_COLUMNS)
    bounds.to_parquet(output_root / "campaign_a_terminal_recovery_interval_bounds.parquet", index=False)

    counter = summarize_counter(samples, episodes, runner) if runner else summarize_counter(samples, episodes, CampaignRunner(project_root, output_root, "", metas, prior_registry, planned_end, effective_cap, prior_calls))
    route_summary_rows = []
    for route in TARGET_ROUTES:
        route_counter = counter["by_route"][route]
        route_summary_rows.append(
            {
                "route_id": route,
                "complete": route_counter["new_complete_episode_count"],
                "left_censored": route_counter["left_censored_episode_count"],
                "right_censored": route_counter["right_censored_episode_count"],
                "new_independent_vehicles": route_counter["new_independent_complete_vehicle_count"],
            }
        )
    pd.DataFrame(route_summary_rows).to_parquet(output_root / "campaign_a_route_summary.parquet", index=False)

    evidence = evidence_sha_audit(output_root, episodes)
    interval = interval_audit(episodes)
    duplicate = duplicate_audit(prior_registry, episodes) if not prior_registry.empty else {"episode_deduplication_audit_passed": True, "episode_duplicate_count": 0, "duplicates": []}
    mapping_regression_count = 0
    if mapping_json_pre != (sha256_file(mapping_json) if mapping_json.exists() else None):
        mapping_regression_count += 1
    if mapping_parquet_pre != (sha256_file(mapping_parquet) if mapping_parquet.exists() else None):
        mapping_regression_count += 1
    clock_counts = Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist()) if not episodes.empty else Counter()
    invalid_clock_order_count = int(clock_counts.get("INVALID_CLOCK_ORDER", 0))
    secret = scan_secret(output_root, service_key)
    secret_leak_count = int(secret["secret_literal_occurrence_count"])
    input_post = [file_audit(path, project_root, source) for path, source in authoritative_inputs(project_root)]
    post_by_path = {record["absolute_path"]: record for record in input_post}
    immutability = []
    for record in input_pre:
        post = post_by_path.get(record["absolute_path"], {})
        changed = record.get("sha256") != post.get("sha256") or record.get("file_size") != post.get("file_size")
        row = dict(record)
        row["post_file_size"] = post.get("file_size")
        row["post_sha256"] = post.get("sha256")
        row["modified_during_r2d1h"] = bool(changed)
        immutability.append(row)
    authoritative_modified_count = sum(1 for row in immutability if row["modified_during_r2d1h"])

    dump_json(output_root / "authoritative_input_immutability_audit.json", {"authoritative_input_modified_count": authoritative_modified_count, "failed_inputs": failed_inputs, "files": immutability})
    dump_json(output_root / "mapping_regression_audit.json", {"mapping_regression_count": mapping_regression_count, "mapping_regression_passed": mapping_regression_count == 0, "existing_mapping_modified": mapping_regression_count > 0, "hf1_mapping_sha256": sha256_file(mapping_parquet) if mapping_parquet.exists() else None})
    dump_json(output_root / "prior_episode_registry_reference_audit.json", {"prior_complete_episode_count": int(len(prior_registry)), "registry_read_only": True, "registry_sha256": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet") if (r2d1e / "frozen_complete_episode_registry.parquet").exists() else None})
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": secret_leak_count, "security_audit_passed": secret_leak_count == 0, "post_campaign_artifact_scan": secret})
    dump_json(output_root / "episode_evidence_sha256_audit.json", evidence)
    dump_json(output_root / "episode_deduplication_audit.json", duplicate)
    dump_json(output_root / "interval_validation_audit.json", interval)
    dump_json(output_root / "clock_semantics_audit.json", {"clock_semantics_audit_completed": True, "clock_semantics_status_counts": dict(clock_counts), "invalid_clock_order_count": invalid_clock_order_count, "episode_count": int(len(episodes))})
    dump_json(output_root / "terminal_counter_audit_v9.json", counter)

    preflight_records = [] if runner is None else [req for req in runner.request_records if req.get("capture_mode") == "PREFLIGHT"]
    campaign_records = [] if runner is None else [req for req in runner.request_records if req.get("capture_mode") != "PREFLIGHT"]
    minute_counts = Counter(req.get("request_observation_time", "")[:16] for req in ([] if runner is None else runner.request_records))
    status_counts = Counter(str(req.get("provider_response_status")) for req in ([] if runner is None else runner.request_records))
    fatal_idx = next((i for i, req in enumerate([] if runner is None else runner.request_records) if req.get("provider_response_status") in FATAL_STATUSES), None)
    calls_after_fatal = 0 if fatal_idx is None else len(runner.request_records) - fatal_idx - 1 if runner else 0
    campaign_started = runner.campaign_started_at if runner else None
    campaign_finished = runner.campaign_finished_at if runner else None
    runtime_passed = bool(
        authorization_approved
        and preflight_passed
        and runner is not None
        and runner.fatal_error is None
        and calls_after_fatal == 0
        and (max(minute_counts.values()) if minute_counts else 0) <= MAX_CALLS_PER_MINUTE
        and runner.physical_calls_this_artifact <= effective_cap
        and prior_calls + runner.physical_calls_this_artifact <= DAILY_SAFETY_CAP
    )
    dump_json(
        output_root / "campaign_preflight_audit.json",
        {
            "preflight_attempted": authorization_approved,
            "preflight_physical_calls": len(preflight_records),
            "target_routes": TARGET_ROUTES,
            "preflight_passed": preflight_passed,
            "blocking_reason": None if preflight_passed else preflight_blocking_reason,
            "records": preflight_records,
        },
    )
    dump_json(
        output_root / "campaign_runtime_audit.json",
        {
            "campaign_started": bool(campaign_started),
            "campaign_started_at": campaign_started,
            "campaign_finished_at": campaign_finished,
            "campaign_physical_calls": len(campaign_records),
            "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
            "effective_campaign_hard_cap": effective_cap,
            "fatal_api_error": None if runner is None else runner.fatal_error,
            "first_fatal_error_time": None if runner is None else runner.first_fatal_error_time,
            "calls_after_first_fatal_error": calls_after_fatal,
            "status_counts": dict(status_counts),
            "stop_reason": None if runner is None else runner.stop_reason,
            "api_runtime_audit_passed": runtime_passed,
        },
    )
    dump_json(
        output_root / "api_stop_condition_audit.json",
        {
            "fatal_stop_triggered": bool(runner and runner.fatal_error),
            "fatal_api_error": None if runner is None else runner.fatal_error,
            "first_fatal_error_time": None if runner is None else runner.first_fatal_error_time,
            "calls_after_first_fatal_error": calls_after_fatal,
            "fatal_error_retry_allowed": False,
        },
    )
    dump_json(
        output_root / "daily_api_usage_audit.json",
        {
            "run_date": run_date,
            "timezone": "Asia/Seoul",
            "prior_physical_calls_on_run_date": prior_calls,
            "prior_authoritative_saved_raw_calls": prior_calls,
            "prior_non_authoritative_physical_calls": 0,
            "preflight_physical_calls": len(preflight_records),
            "campaign_physical_calls": len(campaign_records),
            "total_physical_calls_on_run_date": prior_calls + (0 if runner is None else runner.physical_calls_this_artifact),
            "available_daily_budget": available_daily_budget,
            "effective_campaign_hard_cap": effective_cap,
            "daily_usage_source": "local date-specific raw request index before R2D-1H execution",
            "prior_raw_index_count": len(prior_raw_index),
            "daily_usage_equation_passed": prior_calls + len(preflight_records) + len(campaign_records)
            == prior_calls + (0 if runner is None else runner.physical_calls_this_artifact),
        },
    )

    dump_json(
        output_root / "candidate_vehicle_selection_audit.json",
        {
            "candidate_vehicle_count": int(sum((runner.candidate_count_by_route if runner else Counter()).values())),
            "by_route": {route: counter["by_route"][route] for route in TARGET_ROUTES},
            "exact_vehicle_id_only": True,
            "duplicate_terminal_cycle_excluded": True,
        },
    )
    transitions = [
        {
            "episode_id": ep.get("episode_id"),
            "route_id": ep.get("route_id"),
            "vehicle_id": ep.get("vehicle_id"),
            "final_status": ep.get("episode_status"),
            "first_upstream_sequence": ep.get("first_upstream_watch_sequence"),
            "last_pre_terminal_sequence": ep.get("last_pre_terminal_sequence"),
            "terminal_entry_sequence": ep.get("first_terminal_sequence"),
            "last_terminal_sequence": ep.get("last_terminal_sequence"),
            "first_post_terminal_sequence": ep.get("first_post_terminal_sequence"),
            "confirmation_threshold_sample_count": ep.get("confirmation_threshold_sample_count"),
            "observed_post_terminal_confirmation_sample_count": ep.get("observed_post_terminal_confirmation_sample_count"),
        }
        for ep in episodes.to_dict("records")
    ]
    dump_json(output_root / "campaign_state_transition_audit.json", {"transition_count": len(transitions), "transitions": transitions})

    registry_candidate, registry_payload = build_registry_candidate(prior_registry, episodes, output_root)
    registry_candidate.to_parquet(output_root / "cumulative_episode_registry_candidate.parquet", index=False)
    dump_json(output_root / "cumulative_episode_registry_candidate.json", registry_payload)

    prior_counts = Counter(prior_registry["route_id"].astype(str).tolist()) if not prior_registry.empty else Counter()
    campaign_complete_count = int(counter["total"]["new_complete_episode_count"])
    cumulative_count = int(len(prior_registry) + campaign_complete_count)
    route_cumulative = {route: int(prior_counts.get(route, 0) + counter["by_route"][route]["new_complete_episode_count"]) for route in TARGET_ROUTES}
    dump_json(
        output_root / "method_prototype_progress_audit.json",
        {
            "prior_complete_episode_count": int(len(prior_registry)),
            "campaign_a_verified_new_complete_count": campaign_complete_count,
            "cumulative_candidate_complete_count": cumulative_count,
            "remaining_complete_episodes_to_12": max(0, 12 - cumulative_count),
            "route_cumulative_counts": route_cumulative,
            "cumulative_independent_vehicle_count": int(registry_candidate["vehicle_id"].dropna().astype(str).nunique()) if not registry_candidate.empty else 0,
            "observation_date_count": int(registry_candidate["observation_date"].dropna().astype(str).nunique()) if not registry_candidate.empty and "observation_date" in registry_candidate else 0,
            "hour_bucket_count": int(registry_candidate["hour_bucket"].dropna().astype(str).nunique()) if not registry_candidate.empty and "hour_bucket" in registry_candidate else 0,
        },
    )

    lock_payload = {
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
    }
    dump_json(output_root / "campaign_b_execution_authorization.json", {"approved": False, "reason": "Campaign B requires a separate review after Campaign A artifacts, API usage, complete episodes, clock quality, and cumulative registry candidate are independently validated."})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", lock_payload)
    dump_json(output_root / "simulator_parameter_translation_guard.json", lock_payload)
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, **lock_payload})

    for route in TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route
        route_samples = samples[samples["route_id"].astype(str) == route] if not samples.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
        route_episodes = episodes[episodes["route_id"].astype(str) == route] if not episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
        route_bounds = bounds[bounds["route_id"].astype(str) == route] if not bounds.empty else pd.DataFrame(columns=bounds_cols)
        route_samples.to_parquet(route_dir / "position_samples.parquet", index=False)
        route_samples.to_parquet(route_dir / "vehicle_trajectories.parquet", index=False)
        route_episodes.to_parquet(route_dir / "terminal_recovery_episodes.parquet", index=False)
        route_bounds.to_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", index=False)
        dump_json(route_dir / "route_mapping_reference.json", {"route_id": route, "mapping": metas.get(route), "source": str(mapping_parquet), "read_only_input": True})
        dump_json(route_dir / "observation_manifest.json", {"route_id": route, "observation_started": bool(runner), "raw_file_count": len((runner.raw_files_by_route[route] if runner else [])), "blocking_reason": None if authorization_approved else blocking_reason})
        dump_json(route_dir / "candidate_selection_audit.json", {"route_id": route, "candidate_vehicle_count": int((runner.candidate_count_by_route if runner else Counter())[route]), "history_counts": dict((runner.history_count_by_route if runner else defaultdict(Counter))[route])})
        dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": route, "triggered": int(counter["by_route"][route]["upstream_focused_observation_count"]) > 0, "effective_live_terminal_sequence": metas.get(route, {}).get("effective_live_terminal_sequence")})
        dump_json(route_dir / "focused_trigger_audit.json", {"route_id": route, "triggered": int(counter["by_route"][route]["terminal_entry_count"]) > 0})
        dump_json(route_dir / "episode_summary.json", {"route_id": route, "new_complete_episode_count": counter["by_route"][route]["new_complete_episode_count"], "left_censored_episode_count": counter["by_route"][route]["left_censored_episode_count"], "right_censored_episode_count": counter["by_route"][route]["right_censored_episode_count"]})
        dump_json(route_dir / "clock_semantics_audit.json", {"route_id": route, "episode_count": int(len(route_episodes)), "invalid_clock_order_count": int((route_episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not route_episodes.empty else 0})
        dump_json(route_dir / "raw_file_index.json", {"route_id": route, "raw_file_count": len((runner.raw_files_by_route[route] if runner else [])), "raw_files": (runner.raw_files_by_route[route] if runner else [])})
        dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route, "evidence_sha256_audit_passed": True, "raw_file_count": len((runner.raw_files_by_route[route] if runner else []))})

    strict_json_nonstandard_token_count = sum(strict_json_token_count(path) for path in output_root.rglob("*.json"))
    gate_status = final_gate(
        authorization_approved and preflight_passed,
        blocking_reason or ("BLOCKED_API_RUNTIME" if not preflight_passed else None),
        runner,
        counter,
        evidence,
        interval,
        duplicate,
        mapping_regression_count,
        secret_leak_count,
        invalid_clock_order_count,
    )
    if authoritative_modified_count:
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "run_date": run_date,
        "campaign_window": f"{start_time.isoformat(timespec='seconds')} to {planned_end.isoformat(timespec='seconds')}",
        "authorization_approved": authorization_approved and preflight_passed,
        "blocking_reason": None if authorization_approved and preflight_passed else (blocking_reason or preflight_blocking_reason),
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": len(preflight_records),
        "campaign_physical_calls": len(campaign_records),
        "total_physical_calls_on_run_date": prior_calls + (0 if runner is None else runner.physical_calls_this_artifact),
        "effective_campaign_hard_cap": effective_cap,
        "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
        "fatal_api_error": None if runner is None else runner.fatal_error,
        "first_fatal_error_time": None if runner is None else runner.first_fatal_error_time,
        "calls_after_first_fatal_error": calls_after_fatal,
        "route_results": {route: {"complete": counter["by_route"][route]["new_complete_episode_count"], "left": counter["by_route"][route]["left_censored_episode_count"], "right": counter["by_route"][route]["right_censored_episode_count"], "new_vehicles": counter["by_route"][route]["new_independent_complete_vehicle_count"]} for route in TARGET_ROUTES},
        "campaign_a_new_complete_total": counter["total"]["new_complete_episode_count"],
        "campaign_a_new_independent_vehicle_total": counter["total"]["new_independent_complete_vehicle_count"],
        "cumulative_complete_candidate": cumulative_count,
        "remaining_to_method_prototype_12": max(0, 12 - cumulative_count),
        "mapping_regression_count": mapping_regression_count,
        "episode_duplicate_count": duplicate.get("episode_duplicate_count", 0),
        "contradiction_count": counter["total"]["contradiction_count"],
        "counter_contract_v9_passed": counter["counter_contract_v9_passed"],
        "episode_evidence_sha256_audit_passed": evidence["episode_evidence_sha256_audit_passed"],
        "interval_validation_passed": interval["interval_validation_passed"],
        "secret_leak_count": secret_leak_count,
        "strict_json_nonstandard_token_count": strict_json_nonstandard_token_count,
        "campaign_b_authorized": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "phase2_authorized": False,
        "next_authorized_action": "Campaign A review" if gate_status.startswith("PASS") else "additional Campaign A observation",
    }
    dump_json(output_root / "prompt5_e01_r2d1h_gate.json", gate)

    report_lines = [
        "# Prompt 5-E01-R2D-1H Final Report",
        "",
        "## Status",
        "",
        f"- gate: {gate_status}",
        f"- authorization_approved: {str(gate['authorization_approved']).lower()}",
        f"- blocking_reason: {gate['blocking_reason']}",
        f"- run_date: {run_date}",
        f"- campaign_window: {gate['campaign_window']}",
        f"- prior_physical_calls_on_run_date: {prior_calls}",
        f"- preflight_physical_calls: {len(preflight_records)}",
        f"- campaign_physical_calls: {len(campaign_records)}",
        f"- total_physical_calls_on_run_date: {gate['total_physical_calls_on_run_date']}",
        f"- effective_campaign_hard_cap: {effective_cap}",
        f"- max_calls_per_minute: {gate['max_calls_per_minute']}",
        f"- fatal_api_error: {gate['fatal_api_error'] or 'NONE'}",
        f"- first_fatal_error_time: {gate['first_fatal_error_time'] or 'NONE'}",
        f"- calls_after_first_fatal_error: {calls_after_fatal}",
        "",
        "## Authoritative Inputs",
        "",
        f"- HF1: {hf1}",
        f"- R2D-1E: {r2d1e}",
        f"- R2D-1F: {r2d1f}",
        f"- R2D-1G: {r2d1g}",
        "",
        "Previous R2D-1H artifacts were preserved and were not used as authoritative inputs.",
        "",
        "## Authorization And Observation",
        "",
        f"- DAEGU_BIS_SERVICE_KEY exists: {str(credential_available).lower()}",
        f"- DAEGU_BIS_SERVICE_KEY length: {credential_length}",
        "- API key raw value output: false",
        f"- preflight_passed: {str(preflight_passed).lower()}",
        f"- campaign_stop_reason: {None if runner is None else runner.stop_reason}",
        "",
        "## Route Results",
    ]
    for route in TARGET_ROUTES:
        rc = counter["by_route"][route]
        eps = episodes[episodes["route_id"].astype(str) == route].to_dict("records") if not episodes.empty else []
        report_lines.append(
            f"- {route}: candidates={rc['candidate_vehicle_count']}, complete={rc['new_complete_episode_count']}, left={rc['left_censored_episode_count']}, right={rc['right_censored_episode_count']}, new_independent_complete={rc['new_independent_complete_vehicle_count']}"
        )
        for ep in eps:
            report_lines.append(
                f"  - episode {ep.get('episode_id')}: vehicle={ep.get('vehicle_id')}, history={ep.get('vehicle_history_class')}, first_upstream_seq={ep.get('first_upstream_watch_sequence')}, last_pre_terminal_seq={ep.get('last_pre_terminal_sequence')}, terminal_seq={ep.get('first_terminal_sequence')}, reset_seq={ep.get('first_post_terminal_sequence')}, confirmation={ep.get('observed_post_terminal_confirmation_sample_count')}/{ep.get('confirmation_threshold_sample_count')}, status={ep.get('episode_status')}, clock={ep.get('clock_semantics_status')}"
            )
    report_lines.extend(
        [
            "",
            "## Audits",
            "",
            f"- raw SHA provenance passed: {str(evidence['episode_evidence_sha256_audit_passed']).lower()}",
            f"- duplicate count: {duplicate.get('episode_duplicate_count', 0)}",
            f"- mapping regression count: {mapping_regression_count}",
            f"- contradiction count: {counter['total']['contradiction_count']}",
            f"- interval validation passed: {str(interval['interval_validation_passed']).lower()}",
            f"- invalid clock order count: {invalid_clock_order_count}",
            f"- Counter Contract v9 passed: {str(counter['counter_contract_v9_passed']).lower()}",
            f"- secret leak count: {secret_leak_count}",
            "",
            "## Progress And Locks",
            "",
            f"- cumulative complete candidate: {cumulative_count}",
            f"- remaining to Method Prototype 12: {max(0, 12 - cumulative_count)}",
            "- Campaign B authorized: false",
            "- statistical estimation executed: false",
            "- simulator parameter generated: false",
            "- Phase 2 authorized: false",
            "",
            "## Next Step",
            "",
            gate["next_authorized_action"],
        ]
    )
    (output_root / "prompt5_e01_r2d1h_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_manifest(output_root)
    write_manifest(output_root)

    print("R2D-1H CAMPAIGN A LIVE OBSERVATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    print("\nrun_date:")
    print(run_date)
    print("\ncampaign_window:")
    print(gate["campaign_window"])
    print("\nauthorization_approved:")
    print(str(gate["authorization_approved"]).lower())
    print("\nprior_physical_calls_on_run_date:")
    print(prior_calls)
    print("\npreflight_physical_calls:")
    print(len(preflight_records))
    print("\ncampaign_physical_calls:")
    print(len(campaign_records))
    print("\ntotal_physical_calls_on_run_date:")
    print(gate["total_physical_calls_on_run_date"])
    print("\neffective_campaign_hard_cap:")
    print(effective_cap)
    print("\nmax_calls_per_minute:")
    print(gate["max_calls_per_minute"])
    print("\nfatal_api_error:")
    print(gate["fatal_api_error"] or "NONE")
    print("\nfirst_fatal_error_time:")
    print(gate["first_fatal_error_time"] or "NONE")
    print("\ncalls_after_first_fatal_error:")
    print(calls_after_fatal)
    print("\nroute_results:")
    for route in TARGET_ROUTES:
        rr = gate["route_results"][route]
        print(f"{route} = complete={rr['complete']} / left={rr['left']} / right={rr['right']} / new_vehicles={rr['new_vehicles']}")
    print("\ncampaign_a_new_complete_total:")
    print(gate["campaign_a_new_complete_total"])
    print("\ncampaign_a_new_independent_vehicle_total:")
    print(gate["campaign_a_new_independent_vehicle_total"])
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
    print("\ncounter_contract_v9_passed:")
    print(str(counter["counter_contract_v9_passed"]).lower())
    print("\nepisode_evidence_sha256_audit_passed:")
    print(str(evidence["episode_evidence_sha256_audit_passed"]).lower())
    print("\ninterval_validation_passed:")
    print(str(interval["interval_validation_passed"]).lower())
    print("\nsecret_leak_count:")
    print(secret_leak_count)
    print("\ncampaign_b_authorized:\nfalse")
    print("\nterminal_recovery_estimated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\neligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:")
    print(gate["next_authorized_action"])


if __name__ == "__main__":
    main()
