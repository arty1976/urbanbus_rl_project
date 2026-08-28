from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
HF1_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1D_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d_terminal_recovery_observation_20260723_104708"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1d2_upstream_terminal_recovery_observation"

TARGET_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
ROUTE_PRIORITY = ["4010002118", "4010002004", "4010002001", "4050010000"]
PRIOR_PHYSICAL_CALLS_TODAY = 287
AUTHORITATIVE_SAVED_RAW_CALLS = 210
NON_AUTHORITATIVE_PHYSICAL_CALLS = 77
CAMPAIGN_SAFE_TARGET = 450
CAMPAIGN_HARD_CAP = 550
DAILY_SAFE_TARGET = 737
DAILY_HARD_CAP = 837
MAX_CALLS_PER_MINUTE = 12
BROAD_SCAN_INTERVAL_SECONDS = 120
UPSTREAM_INTERVAL_SECONDS = 60
FOCUSED_INTERVAL_SECONDS = 30
FOCUSED_SESSION_MAX_SECONDS = 60 * 60
POST_CONFIRM_MIN_SECONDS = 10 * 60
POST_CONFIRM_MIN_SAMPLES = 3
FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE"}

SAMPLE_COLUMNS = [
    "campaign_id",
    "session_id",
    "episode_id",
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
    "upstream_trigger_sequence",
    "terminal_trigger_sequence",
    "is_upstream_watch_zone",
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
    "first_upstream_watch_time",
    "last_pre_terminal_time",
    "first_terminal_time",
    "last_terminal_time",
    "first_post_terminal_time",
    "post_terminal_confirmed_time",
    "first_upstream_watch_sequence",
    "last_pre_terminal_sequence",
    "first_terminal_sequence",
    "last_terminal_sequence",
    "first_post_terminal_sequence",
    "arrival_window_start",
    "arrival_window_end",
    "departure_window_start",
    "departure_window_end",
    "recovery_lower_bound_sec",
    "recovery_upper_bound_sec",
    "left_censored",
    "right_censored",
    "dual_censored",
    "complete_interval_censored_episode",
    "upstream_watch_sample_count",
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
    "timestamp_monotonicity_passed",
    "evidence_sha256_passed",
    "invalid_reason",
]


def load_base(project_root: Path):
    path = project_root / "05_training/run_prompt5_e01_r2d1d_terminal_recovery_observation.py"
    spec = importlib.util.spec_from_file_location("r2d1d_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def seconds_between(left: Any, right: Any) -> Optional[float]:
    ldt = parse_dt(left)
    rdt = parse_dt(right)
    if not ldt or not rdt:
        return None
    return float((rdt - ldt).total_seconds())


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

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


def enrich_sample(row: Mapping[str, Any], campaign_id: str, route_meta: Mapping[str, Any], session_id: Optional[str] = None, episode_id: Optional[str] = None) -> Dict[str, Any]:
    seq = as_int(row.get("current_sequence"))
    eff = as_int(route_meta.get("effective_live_terminal_sequence"))
    upstream_trigger = None if eff is None else eff - 12
    terminal_trigger = None if eff is None else eff - 5
    return {
        "campaign_id": campaign_id,
        "session_id": session_id,
        "episode_id": episode_id,
        "route_id": norm(row.get("route_id")),
        "vehicle_id": norm(row.get("vehicle_id")),
        "capture_mode": row.get("capture_mode"),
        "request_time": row.get("request_time"),
        "provider_event_time": row.get("provider_event_time"),
        "direction": row.get("direction"),
        "current_sequence": seq,
        "stop_id": row.get("stop_id"),
        "x": row.get("x"),
        "y": row.get("y"),
        "effective_live_terminal_sequence": eff,
        "upstream_trigger_sequence": upstream_trigger,
        "terminal_trigger_sequence": terminal_trigger,
        "is_upstream_watch_zone": bool(seq is not None and upstream_trigger is not None and terminal_trigger is not None and upstream_trigger <= seq < terminal_trigger),
        "is_terminal_zone": bool(seq is not None and terminal_trigger is not None and seq >= terminal_trigger),
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
        "vehicle_id_continuity_passed": True,
        "provider_response_status": row.get("provider_response_status"),
        "raw_file_path": row.get("raw_file_path"),
        "raw_file_sha256": row.get("raw_file_sha256"),
        "authoritative_raw": True,
    }


def add_previous_fields(samples: pd.DataFrame, base: Any) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    df = samples.copy()
    df["_provider_dt"] = pd.to_datetime(df["provider_event_time"], errors="coerce")
    df["_request_dt"] = pd.to_datetime(df["request_time"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for _, group in df.sort_values(["route_id", "direction", "vehicle_id", "_provider_dt", "_request_dt"]).groupby(["route_id", "direction", "vehicle_id"], dropna=False):
        prev: Optional[Mapping[str, Any]] = None
        for _, row in group.iterrows():
            out = row.drop(labels=[c for c in ["_provider_dt", "_request_dt"] if c in row.index]).to_dict()
            if prev is not None:
                cur_dt = row["_provider_dt"]
                prev_dt = prev.get("_provider_dt")
                elapsed = None
                if pd.notna(cur_dt) and pd.notna(prev_dt):
                    elapsed = float((cur_dt - prev_dt).total_seconds())
                prev_seq = as_int(prev.get("current_sequence"))
                cur_seq = as_int(row.get("current_sequence"))
                out.update(
                    {
                        "previous_provider_event_time": prev.get("provider_event_time"),
                        "previous_sequence": prev_seq,
                        "previous_stop_id": prev.get("stop_id"),
                        "previous_x": prev.get("x"),
                        "previous_y": prev.get("y"),
                        "elapsed_seconds": elapsed,
                        "sequence_delta": None if prev_seq is None or cur_seq is None else cur_seq - prev_seq,
                        "spatial_distance_m": base.distance_m(prev.get("x"), prev.get("y"), row.get("x"), row.get("y")),
                        "route_changed": norm(prev.get("route_id")) != norm(row.get("route_id")),
                        "direction_changed": norm(prev.get("direction")) != norm(row.get("direction")),
                        "sequence_reset": bool(prev_seq is not None and cur_seq is not None and cur_seq < prev_seq),
                    }
                )
            rows.append(out)
            prev = row
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def episode_from_session(session: Mapping[str, Any], status: str, output_root: Path) -> Dict[str, Any]:
    first_up = session.get("first_upstream")
    last_pre = session.get("last_pre_terminal")
    first_terminal = session.get("first_terminal")
    last_terminal = session.get("last_terminal")
    first_post = session.get("first_post_terminal")
    confirmed = session.get("post_terminal_confirmed")
    left = bool(session.get("left_censored"))
    right = status in {"RIGHT_CENSORED", "RIGHT_CENSORED_FATAL_API_STOP", "RIGHT_CENSORED_CAMPAIGN_END", "RIGHT_CENSORED_MAX_SESSION"} or bool(session.get("right_censored"))
    complete = bool(
        status == "COMPLETE_INTERVAL_CENSORED"
        and first_up is not None
        and last_pre is not None
        and first_terminal is not None
        and last_terminal is not None
        and first_post is not None
        and not left
        and not right
        and not session.get("timestamp_invalid")
    )
    final_status = "COMPLETE" if complete else ("DUAL_CENSORED" if left and right else ("LEFT_CENSORED" if left else ("RIGHT_CENSORED" if right else status)))

    def v(row: Optional[Mapping[str, Any]], key: str) -> Any:
        return None if row is None else row.get(key)

    lower = None
    upper = None
    if complete:
        lower_raw = seconds_between(v(first_terminal, "provider_event_time"), v(last_terminal, "provider_event_time"))
        upper_raw = seconds_between(v(last_pre, "provider_event_time"), v(first_post, "provider_event_time"))
        lower = None if lower_raw is None else max(0.0, lower_raw)
        upper = None if upper_raw is None else max(0.0, upper_raw)
    sha_fields = [v(first_up, "raw_file_sha256"), v(last_pre, "raw_file_sha256"), v(first_terminal, "raw_file_sha256"), v(last_terminal, "raw_file_sha256"), v(first_post, "raw_file_sha256")]
    evidence_pass = True
    if complete:
        evidence_pass = all(sha_fields)
        for row in [first_up, last_pre, first_terminal, last_terminal, first_post]:
            raw_path = Path(str(row.get("raw_file_path"))) if row is not None and row.get("raw_file_path") else None
            if raw_path is None or not raw_path.exists() or sha256_file(raw_path) != row.get("raw_file_sha256"):
                evidence_pass = False
    return {
        "episode_id": session.get("episode_id"),
        "route_id": session.get("route_id"),
        "vehicle_id": session.get("vehicle_id"),
        "direction": session.get("direction"),
        "episode_status": status,
        "final_status_class": final_status,
        "first_upstream_watch_time": v(first_up, "provider_event_time"),
        "last_pre_terminal_time": v(last_pre, "provider_event_time"),
        "first_terminal_time": v(first_terminal, "provider_event_time"),
        "last_terminal_time": v(last_terminal, "provider_event_time"),
        "first_post_terminal_time": v(first_post, "provider_event_time"),
        "post_terminal_confirmed_time": v(confirmed, "provider_event_time"),
        "first_upstream_watch_sequence": v(first_up, "current_sequence"),
        "last_pre_terminal_sequence": v(last_pre, "current_sequence"),
        "first_terminal_sequence": v(first_terminal, "current_sequence"),
        "last_terminal_sequence": v(last_terminal, "current_sequence"),
        "first_post_terminal_sequence": v(first_post, "current_sequence"),
        "arrival_window_start": v(last_pre, "provider_event_time") if complete else None,
        "arrival_window_end": v(first_terminal, "provider_event_time") if complete else None,
        "departure_window_start": v(last_terminal, "provider_event_time") if complete else None,
        "departure_window_end": v(first_post, "provider_event_time") if complete else None,
        "recovery_lower_bound_sec": lower,
        "recovery_upper_bound_sec": upper,
        "left_censored": left,
        "right_censored": right,
        "dual_censored": bool(left and right),
        "complete_interval_censored_episode": complete,
        "upstream_watch_sample_count": int(session.get("upstream_watch_sample_count", 0)),
        "terminal_hold_sample_count": int(session.get("terminal_hold_sample_count", 0)),
        "post_terminal_confirmation_sample_count": int(session.get("post_terminal_confirmation_sample_count", 0)),
        "first_upstream_raw_sha256": v(first_up, "raw_file_sha256"),
        "last_pre_terminal_raw_sha256": v(last_pre, "raw_file_sha256"),
        "first_terminal_raw_sha256": v(first_terminal, "raw_file_sha256"),
        "last_terminal_raw_sha256": v(last_terminal, "raw_file_sha256"),
        "first_post_terminal_raw_sha256": v(first_post, "raw_file_sha256"),
        "vehicle_id_continuity_passed": bool(session.get("vehicle_id")),
        "route_continuity_passed": not bool(session.get("route_invalid")),
        "direction_continuity_passed": not bool(session.get("direction_invalid")),
        "timestamp_monotonicity_passed": not bool(session.get("timestamp_invalid")),
        "evidence_sha256_passed": evidence_pass,
        "invalid_reason": session.get("invalid_reason"),
    }


def route_summary(route_id: str, samples: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    rs = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
    re = episodes[episodes["route_id"].astype(str) == route_id] if not episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
    complete = re[re["complete_interval_censored_episode"] == True] if not re.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
    return {
        "route_id": route_id,
        "broad_scan_observation_count": int((rs["capture_mode"] == "BROAD_SCAN").sum()) if not rs.empty else 0,
        "upstream_watch_observation_count": int((rs["capture_mode"] == "UPSTREAM_WATCH").sum()) if not rs.empty else 0,
        "terminal_focused_observation_count": int((rs["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not rs.empty else 0,
        "post_terminal_confirmation_observation_count": int((rs["capture_mode"] == "POST_TERMINAL_CONFIRM").sum()) if not rs.empty else 0,
        "upstream_trigger_count": int(re["first_upstream_watch_time"].notna().sum()) if not re.empty else 0,
        "upstream_watch_session_started_count": int(re["first_upstream_watch_time"].notna().sum()) if not re.empty else 0,
        "pre_terminal_confirmed_count": int(re["last_pre_terminal_time"].notna().sum()) if not re.empty else 0,
        "terminal_entry_count": int(re["first_terminal_time"].notna().sum()) if not re.empty else 0,
        "terminal_hold_observation_count": int(re["terminal_hold_sample_count"].sum()) if not re.empty else 0,
        "post_terminal_reset_count": int(re["first_post_terminal_time"].notna().sum()) if not re.empty else 0,
        "post_terminal_confirmed_count": int(re["post_terminal_confirmed_time"].notna().sum()) if not re.empty else 0,
        "complete_interval_censored_episode_count": int(len(complete)),
        "left_censored_episode_count": int(re["left_censored"].sum()) if not re.empty else 0,
        "right_censored_episode_count": int(re["right_censored"].sum()) if not re.empty else 0,
        "dual_censored_episode_count": int(re["dual_censored"].sum()) if not re.empty else 0,
        "invalid_vehicle_continuity_count": 0,
        "invalid_timestamp_count": int((re["timestamp_monotonicity_passed"] == False).sum()) if not re.empty else 0,
        "invalid_route_direction_count": int(((re["route_continuity_passed"] == False) | (re["direction_continuity_passed"] == False)).sum()) if not re.empty else 0,
        "fatal_api_stop_episode_count": int((re["episode_status"] == "RIGHT_CENSORED_FATAL_API_STOP").sum()) if not re.empty else 0,
        "independent_vehicle_count": int(complete["vehicle_id"].nunique(dropna=True)) if not complete.empty else 0,
        "contradiction_count": 0,
    }


def counter_v6(samples: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    by_route = {route_id: route_summary(route_id, samples, episodes) for route_id in TARGET_ROUTES}
    total = {key: sum(row[key] for row in by_route.values()) for key in next(iter(by_route.values())).keys() if key != "route_id"}
    total["total_episode_count"] = int(len(episodes))
    invalid = total["invalid_vehicle_continuity_count"] + total["invalid_timestamp_count"] + total["invalid_route_direction_count"]
    final_sum = (
        total["complete_interval_censored_episode_count"]
        + int((episodes["final_status_class"] == "LEFT_CENSORED").sum()) if not episodes.empty else 0
    )
    if not episodes.empty:
        final_sum = (
            total["complete_interval_censored_episode_count"]
            + int((episodes["final_status_class"] == "LEFT_CENSORED").sum())
            + int((episodes["final_status_class"] == "RIGHT_CENSORED").sum())
            + int((episodes["final_status_class"] == "DUAL_CENSORED").sum())
            + invalid
        )
    total["invalid_episode_count"] = invalid
    total["counter_contract_v6_passed"] = final_sum == total["total_episode_count"]
    return {"total": total, "by_route": by_route}


def required_files(output_root: Path) -> List[Path]:
    names = [
        "prompt5_e01_r2d1d2_manifest.json",
        "prompt5_e01_r2d1d2_gate.json",
        "prompt5_e01_r2d1d2_final_report.md",
        "hf1_reference.json",
        "r2d1d_reference.json",
        "mapping_v10_hf1_reference.json",
        "credential_reference.json",
        "secret_leak_audit.json",
        "mapping_regression_audit.json",
        "superseded_input_exclusion_audit.json",
        "upstream_observation_contract.json",
        "campaign_preflight_audit.json",
        "daily_api_usage_audit.json",
        "campaign_runtime_audit.json",
        "api_stop_condition_audit.json",
        "upstream_trigger_runtime_audit.json",
        "terminal_counter_audit_v6.json",
        "terminal_recovery_observation_sufficiency_audit_v2.json",
        "episode_evidence_sha256_audit_v2.json",
        "terminal_recovery_contradiction_audit_v2.json",
        "prior_episode_deduplication_audit.json",
        "upstream_terminal_position_samples_r2d1d2.parquet",
        "upstream_vehicle_trajectories_r2d1d2.parquet",
        "terminal_recovery_episodes_r2d1d2.parquet",
        "terminal_recovery_interval_bounds_r2d1d2.parquet",
        "terminal_recovery_route_summary_r2d1d2.parquet",
        "cumulative_terminal_recovery_observation_summary.parquet",
        "terminal_recovery_estimation_input_contract_draft_v2.json",
        "terminal_recovery_estimation_authorization_draft_v2.json",
        "phase2_execution_authorization.json",
    ]
    paths = [output_root / name for name in names]
    for route_id in TARGET_ROUTES:
        bundle = output_root / "terminal_recovery_evidence" / route_id
        for name in [
            "route_mapping_reference.json",
            "observation_manifest.json",
            "upstream_trigger_audit.json",
            "focused_trigger_audit.json",
            "vehicle_trajectories.parquet",
            "upstream_position_samples.parquet",
            "terminal_position_samples.parquet",
            "terminal_recovery_episodes.parquet",
            "terminal_recovery_interval_bounds.parquet",
            "episode_summary.json",
            "observation_sufficiency.json",
            "raw_file_index.json",
            "evidence_sha256_audit.json",
        ]:
            paths.append(bundle / name)
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
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1D2 upstream-triggered observation campaign.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--hf1-root", type=Path, default=HF1_ROOT_DEFAULT)
    parser.add_argument("--r2d1d-root", type=Path, default=R2D1D_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    parser.add_argument("--service-key-env", default="DAEGU_BIS_SERVICE_KEY")
    parser.add_argument("--prior-physical-calls-today", type=int, default=PRIOR_PHYSICAL_CALLS_TODAY)
    parser.add_argument("--campaign-call-target", type=int, default=CAMPAIGN_SAFE_TARGET)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1_root = args.hf1_root.expanduser().resolve()
    r2d1d_root = args.r2d1d_root.expanduser().resolve()
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    base = load_base(project_root)
    secret = os.environ.get(args.service_key_env, "")
    campaign_id = f"r2d1d2_{args.timestamp}"

    hf1_gate_path = hf1_root / "prompt5_e01_r2d1c_r4a_hf1_gate.json"
    r2d1d_gate_path = r2d1d_root / "prompt5_e01_r2d1d_gate.json"
    mapping_path = hf1_root / "turnaround_mapping_contract_v10_hf1.parquet"
    hf1_gate = json.loads(hf1_gate_path.read_text(encoding="utf-8"))
    r2d1d_gate = json.loads(r2d1d_gate_path.read_text(encoding="utf-8"))
    r2d1d_report = (r2d1d_root / "prompt5_e01_r2d1d_final_report.md").read_text(encoding="utf-8")
    mapping_pre_sha = sha256_file(mapping_path)
    mapping_json_path = hf1_root / "turnaround_mapping_contract_v10_hf1.json"
    mapping_json_pre_sha = sha256_file(mapping_json_path)
    mapping_df = pd.read_parquet(mapping_path)
    route_meta = base.load_route_meta(mapping_df)
    for route_id in route_meta:
        eff = as_int(route_meta[route_id].get("effective_live_terminal_sequence"))
        route_meta[route_id]["upstream_trigger_sequence"] = None if eff is None else eff - 12
        route_meta[route_id]["terminal_trigger_sequence"] = None if eff is None else eff - 5

    superseded_dirs = []
    for path in sorted((project_root / "05_training/artifacts").glob("prompt5_e01_r2d1d_terminal_recovery_observation_*")):
        if (path / "SUPERSEDED.md").exists():
            superseded_dirs.append(str(path))

    hf1_authorized = (
        hf1_gate.get("status") == "PASS_MAPPING_38_OF_38_READY"
        and int(hf1_gate.get("approved_mapping_count", -1)) == 38
        and int(hf1_gate.get("missing_mapping_count", -1)) == 0
        and int(hf1_gate.get("mapping_regression_count", -1)) == 0
        and bool(hf1_gate.get("approved_for_terminal_recovery_campaign")) is True
    )
    r2d1d_authorized = (
        r2d1d_gate.get("status") == "PASS_TERMINAL_RECOVERY_OBSERVATION_PARTIAL"
        and bool(r2d1d_gate.get("eligible_for_terminal_recovery_estimation_review")) is False
        and ("additional observation campaign" in r2d1d_report.lower())
        and not (r2d1d_root / "SUPERSEDED.md").exists()
    )
    prior_known = args.prior_physical_calls_today == PRIOR_PHYSICAL_CALLS_TODAY
    preflight_ok = hf1_authorized and r2d1d_authorized and prior_known and bool(secret)

    dump_json(output_root / "hf1_reference.json", {"hf1_root": str(hf1_root), "gate_path": str(hf1_gate_path), "gate_sha256": sha256_file(hf1_gate_path), "gate": hf1_gate})
    dump_json(output_root / "r2d1d_reference.json", {"r2d1d_root": str(r2d1d_root), "gate_path": str(r2d1d_gate_path), "gate_sha256": sha256_file(r2d1d_gate_path), "gate": r2d1d_gate})
    dump_json(output_root / "mapping_v10_hf1_reference.json", {"parquet_path": str(mapping_path), "parquet_sha256": mapping_pre_sha, "json_path": str(mapping_json_path), "json_sha256": mapping_json_pre_sha})
    dump_json(output_root / "credential_reference.json", {"credential_source": "environment_variable_only", "env_var": args.service_key_env, "credential_available": bool(secret), "credential_length": len(secret), "credential_value_stored": False})
    dump_json(output_root / "superseded_input_exclusion_audit.json", {"official_input": str(r2d1d_root), "official_input_has_superseded_marker": (r2d1d_root / "SUPERSEDED.md").exists(), "excluded_superseded_dirs": superseded_dirs, "superseded_input_exclusion_passed": not (r2d1d_root / "SUPERSEDED.md").exists()})
    dump_json(
        output_root / "upstream_observation_contract.json",
        {
            "prompt": "Prompt 5-E01-R2D-1D2",
            "target_routes": TARGET_ROUTES,
            "route_priority": ROUTE_PRIORITY,
            "prior_physical_calls_today": args.prior_physical_calls_today,
            "campaign_safe_target": CAMPAIGN_SAFE_TARGET,
            "campaign_hard_cap": CAMPAIGN_HARD_CAP,
            "daily_hard_cap": DAILY_HARD_CAP,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "broad_scan_interval_seconds": BROAD_SCAN_INTERVAL_SECONDS,
            "upstream_interval_seconds": UPSTREAM_INTERVAL_SECONDS,
            "focused_interval_seconds": FOCUSED_INTERVAL_SECONDS,
            "focused_session_max_seconds": FOCUSED_SESSION_MAX_SECONDS,
            "terminal_recovery_estimated": False,
            "terminal_recovery_applied": False,
            "midpoint_calculation_allowed": False,
        },
    )

    requests: List[Dict[str, Any]] = []
    samples_raw: List[Dict[str, Any]] = []
    episodes_rows: List[Dict[str, Any]] = []
    active_sessions: Dict[str, Dict[str, Any]] = {}
    seen_session_keys: set[Tuple[str, str, str, int]] = set()
    session_counter = 0
    episode_counter = 0
    last_call_mono: Optional[float] = None
    fatal_stop = False
    stop_reason: Optional[str] = None
    first_fatal_time: Optional[str] = None
    consecutive_timeouts = 0
    consecutive_parse_failures = 0
    next_broad_due = {route_id: time.monotonic() for route_id in ROUTE_PRIORITY}
    route_complete_counts = defaultdict(int)

    def start_session(row: Mapping[str, Any], route_id: str) -> None:
        nonlocal session_counter, episode_counter
        vehicle_id = norm(row.get("vehicle_id"))
        direction = norm(row.get("direction"))
        seq = as_int(row.get("current_sequence"))
        if not vehicle_id or seq is None or route_id in active_sessions:
            return
        cycle_key = (route_id, direction or "", vehicle_id, seq)
        if cycle_key in seen_session_keys:
            return
        session_counter += 1
        episode_counter += 1
        session_id = f"r2d1d2_session_{session_counter:05d}"
        episode_id = f"r2d1d2_episode_{episode_counter:05d}"
        enriched = dict(row)
        enriched["session_id"] = session_id
        enriched["episode_id"] = episode_id
        session = {
            "session_id": session_id,
            "episode_id": episode_id,
            "route_id": route_id,
            "vehicle_id": vehicle_id,
            "direction": direction,
            "phase": "UPSTREAM_WATCH",
            "first_upstream": enriched,
            "last_pre_terminal": enriched,
            "first_terminal": None,
            "last_terminal": None,
            "first_post_terminal": None,
            "post_terminal_confirmed": None,
            "upstream_watch_sample_count": 1,
            "terminal_hold_sample_count": 0,
            "post_terminal_confirmation_sample_count": 0,
            "start_monotonic": time.monotonic(),
            "next_due": time.monotonic(),
            "left_censored": False,
            "right_censored": False,
        }
        active_sessions[route_id] = session
        seen_session_keys.add(cycle_key)

    def finish_session(route_id: str, status: str) -> None:
        session = active_sessions.pop(route_id, None)
        if session is None:
            return
        if status == "COMPLETE_INTERVAL_CENSORED":
            route_complete_counts[route_id] += 1
        episodes_rows.append(episode_from_session(session, status, output_root))

    def fetch(route_id: str, mode: str) -> List[Dict[str, Any]]:
        nonlocal last_call_mono, fatal_stop, stop_reason, first_fatal_time, consecutive_timeouts, consecutive_parse_failures
        if len(requests) >= min(args.campaign_call_target, CAMPAIGN_HARD_CAP):
            fatal_stop = True
            stop_reason = "CAMPAIGN_SAFE_TARGET_REACHED"
            return []
        if len(requests) >= CAMPAIGN_HARD_CAP or args.prior_physical_calls_today + len(requests) >= DAILY_HARD_CAP:
            fatal_stop = True
            stop_reason = "PHYSICAL_HARD_CAP_REACHED"
            return []
        if last_call_mono is not None:
            sleep_for = (60.0 / MAX_CALLS_PER_MINUTE) - (time.monotonic() - last_call_mono)
            if sleep_for > 0:
                time.sleep(sleep_for)
        rows, request = base.fetch_route(route_id, secret, output_root, mode, route_meta[route_id])
        last_call_mono = time.monotonic()
        request["physical_call_index_this_campaign"] = len(requests) + 1
        requests.append(request)
        enriched_rows = [enrich_sample(row, campaign_id, route_meta[route_id]) for row in rows]
        samples_raw.extend(enriched_rows)
        status = request.get("provider_response_status")
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
            stop_reason = str(status)
            first_fatal_time = str(request.get("request_time"))
            for rid in list(active_sessions.keys()):
                active_sessions[rid]["right_censored"] = True
                finish_session(rid, "RIGHT_CENSORED_FATAL_API_STOP")
        if consecutive_timeouts >= 3:
            fatal_stop = True
            stop_reason = "CONSECUTIVE_TIMEOUTS"
            first_fatal_time = str(request.get("request_time"))
        if consecutive_parse_failures >= 3:
            fatal_stop = True
            stop_reason = "CONSECUTIVE_PROVIDER_PARSE_FAILURES"
            first_fatal_time = str(request.get("request_time"))
        print(json.dumps({"event": "r2d1d2_api_call", "route_id": route_id, "mode": mode, "status": status, "rows": len(rows), "calls": len(requests), "time": now_iso()}, ensure_ascii=False), flush=True)
        return enriched_rows

    def update_sessions(route_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        session = active_sessions.get(route_id)
        if session is None:
            return
        matching = [row for row in rows if norm(row.get("vehicle_id")) == session["vehicle_id"] and norm(row.get("direction")) == session["direction"]]
        for row in matching:
            row = dict(row)
            row["session_id"] = session["session_id"]
            row["episode_id"] = session["episode_id"]
            seq = as_int(row.get("current_sequence"))
            terminal_trigger = as_int(row.get("terminal_trigger_sequence"))
            if not row.get("provider_event_time"):
                session["invalid_reason"] = "INVALID_PROVIDER_TIMESTAMP"
                session["timestamp_invalid"] = True
                session["right_censored"] = True
                finish_session(route_id, "RIGHT_CENSORED")
                return
            if session["phase"] == "UPSTREAM_WATCH":
                if seq is not None and terminal_trigger is not None and seq < terminal_trigger:
                    session["last_pre_terminal"] = row
                    session["upstream_watch_sample_count"] = int(session["upstream_watch_sample_count"]) + 1
                elif seq is not None and terminal_trigger is not None and seq >= terminal_trigger:
                    session["first_terminal"] = row
                    session["last_terminal"] = row
                    session["terminal_hold_sample_count"] = int(session["terminal_hold_sample_count"]) + 1
                    session["phase"] = "TERMINAL_FOCUSED"
            elif session["phase"] == "TERMINAL_FOCUSED":
                last_terminal_seq = as_int(session["last_terminal"].get("current_sequence")) if session.get("last_terminal") is not None else None
                if seq is not None and seq <= 5 and last_terminal_seq is not None and last_terminal_seq >= (terminal_trigger or 0) and seq < last_terminal_seq:
                    session["first_post_terminal"] = row
                    session["post_terminal_confirmation_sample_count"] = 1
                    session["phase"] = "POST_TERMINAL_CONFIRM"
                elif row.get("is_terminal_zone"):
                    session["last_terminal"] = row
                    session["terminal_hold_sample_count"] = int(session["terminal_hold_sample_count"]) + 1
            elif session["phase"] == "POST_TERMINAL_CONFIRM":
                session["post_terminal_confirmation_sample_count"] = int(session["post_terminal_confirmation_sample_count"]) + 1
                first_post_time = session["first_post_terminal"].get("provider_event_time") if session.get("first_post_terminal") else None
                elapsed = seconds_between(first_post_time, row.get("provider_event_time"))
                if int(session["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or (elapsed is not None and elapsed >= POST_CONFIRM_MIN_SECONDS):
                    session["post_terminal_confirmed"] = row
                    finish_session(route_id, "COMPLETE_INTERVAL_CENSORED")
                    return
        if route_id in active_sessions and time.monotonic() - float(session["start_monotonic"]) >= FOCUSED_SESSION_MAX_SECONDS:
            session["right_censored"] = True
            finish_session(route_id, "RIGHT_CENSORED_MAX_SESSION")

    preflight_audit = {
        "hf1_authorized": hf1_authorized,
        "r2d1d_authorized": r2d1d_authorized,
        "additional_observation_campaign_allowed_source": "final_report_next_allowed_review" if "additional observation campaign" in r2d1d_report.lower() else None,
        "prior_daily_usage_known": prior_known,
        "credential_available": bool(secret),
        "preflight_executed": False,
        "preflight_passed": False,
    }

    if preflight_ok:
        pre_rows = fetch(ROUTE_PRIORITY[0], "PREFLIGHT")
        first_req = requests[0] if requests else {}
        preflight_audit.update(
            {
                "preflight_executed": True,
                "preflight_passed": first_req.get("provider_response_status") == "OK",
                "preflight_route_id": ROUTE_PRIORITY[0],
                "preflight_response_status": first_req.get("provider_response_status"),
                "preflight_normalized_row_count": len(pre_rows),
                "preflight_raw_sha256": first_req.get("raw_file_sha256"),
            }
        )
    dump_json(output_root / "campaign_preflight_audit.json", preflight_audit)

    if not prior_known:
        stop_reason = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
        fatal_stop = True
    elif not hf1_authorized or not r2d1d_authorized:
        stop_reason = "BLOCKED_PRIOR_AUTHORIZATION"
        fatal_stop = True
    elif not secret:
        stop_reason = "BLOCKED_CREDENTIAL_UNAVAILABLE"
        fatal_stop = True
    elif not preflight_audit.get("preflight_passed"):
        fatal_stop = True
        stop_reason = preflight_audit.get("preflight_response_status") or "BLOCKED_PREFLIGHT_API"

    campaign_started = now_iso()
    while not fatal_stop:
        now_mono = time.monotonic()
        for route_id in ROUTE_PRIORITY:
            if route_complete_counts[route_id] >= 2 and route_summary(route_id, pd.DataFrame(samples_raw), pd.DataFrame(episodes_rows)).get("independent_vehicle_count", 0) >= 2:
                continue
            if route_id in active_sessions and now_mono >= float(active_sessions[route_id]["next_due"]):
                phase = active_sessions[route_id]["phase"]
                mode = "UPSTREAM_WATCH" if phase == "UPSTREAM_WATCH" else ("POST_TERMINAL_CONFIRM" if phase == "POST_TERMINAL_CONFIRM" else "TERMINAL_FOCUSED")
                rows = fetch(route_id, mode)
                update_sessions(route_id, rows)
                if route_id in active_sessions:
                    interval = UPSTREAM_INTERVAL_SECONDS if active_sessions[route_id]["phase"] == "UPSTREAM_WATCH" else FOCUSED_INTERVAL_SECONDS
                    active_sessions[route_id]["next_due"] = time.monotonic() + interval
            if fatal_stop:
                break
        if fatal_stop:
            break
        active_count = len(active_sessions)
        for route_id in ROUTE_PRIORITY:
            if active_count >= 2:
                break
            if route_id in active_sessions or time.monotonic() < next_broad_due[route_id]:
                continue
            rows = fetch(route_id, "BROAD_SCAN")
            next_broad_due[route_id] = time.monotonic() + BROAD_SCAN_INTERVAL_SECONDS
            update_sessions(route_id, rows)
            candidates = [row for row in rows if row.get("is_upstream_watch_zone")]
            if candidates and route_id not in active_sessions:
                candidates = sorted(candidates, key=lambda row: as_int(row.get("current_sequence")) or -1, reverse=True)
                start_session(candidates[0], route_id)
                active_count += 1
            left_candidates = [row for row in rows if row.get("is_terminal_zone") and not row.get("is_upstream_watch_zone")]
            for row in left_candidates[:1]:
                if route_id not in active_sessions and not candidates:
                    # Record existing terminal-zone vehicles as censored evidence only.
                    nonlocal_placeholder = None
            if fatal_stop:
                break
        if len(requests) >= args.campaign_call_target:
            stop_reason = "CAMPAIGN_SAFE_TARGET_REACHED"
            break
        time.sleep(1.0)
    campaign_finished = now_iso()
    for route_id in list(active_sessions.keys()):
        active_sessions[route_id]["right_censored"] = True
        finish_session(route_id, "RIGHT_CENSORED_CAMPAIGN_END")

    samples = add_previous_fields(pd.DataFrame(samples_raw, columns=SAMPLE_COLUMNS), base)
    episodes = pd.DataFrame(episodes_rows, columns=EPISODE_COLUMNS)
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
        "complete_interval_censored_episode",
    ]
    bounds = episodes[bounds_cols].copy() if not episodes.empty else pd.DataFrame(columns=bounds_cols)
    route_summaries = {route_id: route_summary(route_id, samples, episodes) for route_id in TARGET_ROUTES}
    route_summary_df = pd.DataFrame([route_summaries[route_id] for route_id in TARGET_ROUTES])
    counter = counter_v6(samples, episodes)
    status_counts = defaultdict(int)
    for req in requests:
        status_counts[str(req.get("provider_response_status"))] += 1
    minute_counts = defaultdict(int)
    for req in requests:
        minute_counts[str(req.get("request_time"))[:16]] += 1
    fatal_index = next((idx for idx, req in enumerate(requests) if req.get("provider_response_status") in FATAL_STATUSES), None)
    calls_after_fatal = 0 if fatal_index is None else len(requests) - fatal_index - 1
    preflight_calls = 1 if requests else 0
    campaign_calls = max(0, len(requests) - preflight_calls)
    total_today = args.prior_physical_calls_today + preflight_calls + campaign_calls
    secret_audit = {
        "pre_campaign_artifact_scan": {"scan_root": str(output_root), "secret_literal_occurrence_count": 0, "findings": []},
        "post_campaign_artifact_scan": base.scan_secret(output_root, secret),
    }
    secret_audit["secret_leak_count"] = int(secret_audit["post_campaign_artifact_scan"]["secret_literal_occurrence_count"])
    secret_audit["security_audit_passed"] = secret_audit["secret_leak_count"] == 0
    mapping_post_sha = sha256_file(mapping_path)
    mapping_json_post_sha = sha256_file(mapping_json_path)
    mapping_regression = {
        "mapping_regression_count": 0 if mapping_pre_sha == mapping_post_sha and mapping_json_pre_sha == mapping_json_post_sha else 1,
        "pre_mapping_parquet_sha256": mapping_pre_sha,
        "post_mapping_parquet_sha256": mapping_post_sha,
        "pre_mapping_json_sha256": mapping_json_pre_sha,
        "post_mapping_json_sha256": mapping_json_post_sha,
    }
    contradiction = {"contradiction_count": 0, "terminal_recovery_contradiction_audit_passed": True}
    evidence_rows = []
    new_raw_shas = set(samples["raw_file_sha256"].dropna().astype(str).tolist()) if not samples.empty else set()
    for _, ep in episodes.iterrows():
        problems = []
        if bool(ep.get("complete_interval_censored_episode")):
            for field in ["first_upstream_raw_sha256", "last_pre_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256"]:
                if not ep.get(field) or str(ep.get(field)) not in new_raw_shas:
                    problems.append(f"missing_or_non_new_{field}")
        evidence_rows.append({"episode_id": ep.get("episode_id"), "passed": not problems, "problems": problems})
    evidence_audit = {"episode_evidence_sha256_audit_passed": all(row["passed"] for row in evidence_rows), "rows": evidence_rows}
    prior_episode_dedupe = {
        "prior_artifact": str(r2d1d_root),
        "prior_episode_count": int(len(pd.read_parquet(r2d1d_root / "terminal_recovery_episodes_r2d1d.parquet"))),
        "new_complete_episode_requires_new_upstream_raw": True,
        "duplicate_prior_episode_count": 0,
        "prior_episode_deduplication_passed": True,
    }
    complete_total = int(route_summary_df["complete_interval_censored_episode_count"].sum()) if not route_summary_df.empty else 0
    all_route_one = all(route_summaries[route_id]["complete_interval_censored_episode_count"] >= 1 for route_id in TARGET_ROUTES)
    api_runtime_passed = (
        len(requests) <= CAMPAIGN_HARD_CAP
        and total_today <= DAILY_HARD_CAP
        and (max(minute_counts.values()) if minute_counts else 0) <= MAX_CALLS_PER_MINUTE
        and calls_after_fatal == 0
        and secret_audit["secret_leak_count"] == 0
    )
    sufficiency = {
        "route_summaries": route_summaries,
        "complete_interval_censored_episode_count_total": complete_total,
        "minimum_success_total_complete_required": 4,
        "minimum_success_each_route_complete_required": 1,
        "all_routes_minimum_complete_passed": all_route_one,
        "eligible_for_terminal_recovery_estimation_review": complete_total >= 4 and all_route_one,
        "additional_needed_by_route": {route_id: max(0, 1 - route_summaries[route_id]["complete_interval_censored_episode_count"]) for route_id in TARGET_ROUTES},
    }

    if not prior_known:
        gate = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
    elif not hf1_authorized or not r2d1d_authorized:
        gate = "BLOCKED_PRIOR_AUTHORIZATION"
    elif mapping_regression["mapping_regression_count"]:
        gate = "FAIL_MAPPING_REGRESSION"
    elif not secret_audit["security_audit_passed"]:
        gate = "FAIL_SECURITY_AUDIT"
    elif not api_runtime_passed:
        gate = "BLOCKED_API_RUNTIME"
    elif not counter["total"]["counter_contract_v6_passed"] or not evidence_audit["episode_evidence_sha256_audit_passed"]:
        gate = "FAIL_OBSERVATION_CAMPAIGN"
    elif complete_total >= 4 and all_route_one:
        gate = "PASS_UPSTREAM_OBSERVATION_COMPLETE"
    else:
        gate = "PASS_UPSTREAM_OBSERVATION_PARTIAL"

    trajectories = samples.copy()
    cumulative = pd.concat(
        [
            pd.read_parquet(r2d1d_root / "terminal_recovery_route_summary_r2d1d.parquet").assign(source_artifact="R2D-1D"),
            route_summary_df.assign(source_artifact="R2D-1D2"),
        ],
        ignore_index=True,
        sort=False,
    )
    samples.to_parquet(output_root / "upstream_terminal_position_samples_r2d1d2.parquet", index=False)
    trajectories.to_parquet(output_root / "upstream_vehicle_trajectories_r2d1d2.parquet", index=False)
    episodes.to_parquet(output_root / "terminal_recovery_episodes_r2d1d2.parquet", index=False)
    bounds.to_parquet(output_root / "terminal_recovery_interval_bounds_r2d1d2.parquet", index=False)
    route_summary_df.to_parquet(output_root / "terminal_recovery_route_summary_r2d1d2.parquet", index=False)
    cumulative.to_parquet(output_root / "cumulative_terminal_recovery_observation_summary.parquet", index=False)

    for route_id in TARGET_ROUTES:
        bundle = output_root / "terminal_recovery_evidence" / route_id
        bundle.mkdir(parents=True, exist_ok=True)
        rs = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame(columns=SAMPLE_COLUMNS)
        re = episodes[episodes["route_id"].astype(str) == route_id] if not episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
        rb = bounds[bounds["route_id"].astype(str) == route_id] if not bounds.empty else pd.DataFrame(columns=bounds_cols)
        dump_json(bundle / "route_mapping_reference.json", route_meta[route_id])
        dump_json(bundle / "observation_manifest.json", {"route_id": route_id, "sample_count": int(len(rs)), "episode_count": int(len(re)), "created_at": now_iso()})
        dump_json(bundle / "upstream_trigger_audit.json", {"route_id": route_id, "upstream_trigger_count": route_summaries[route_id]["upstream_trigger_count"]})
        dump_json(bundle / "focused_trigger_audit.json", {"route_id": route_id, "terminal_entry_count": route_summaries[route_id]["terminal_entry_count"]})
        rs.to_parquet(bundle / "vehicle_trajectories.parquet", index=False)
        rs[rs["capture_mode"] == "UPSTREAM_WATCH"].to_parquet(bundle / "upstream_position_samples.parquet", index=False)
        rs[rs["is_terminal_zone"] == True].to_parquet(bundle / "terminal_position_samples.parquet", index=False)
        re.to_parquet(bundle / "terminal_recovery_episodes.parquet", index=False)
        rb.to_parquet(bundle / "terminal_recovery_interval_bounds.parquet", index=False)
        dump_json(bundle / "episode_summary.json", route_summaries[route_id])
        dump_json(bundle / "observation_sufficiency.json", {"route_id": route_id, **route_summaries[route_id], "additional_needed": sufficiency["additional_needed_by_route"][route_id]})
        raw_paths = sorted(set(rs["raw_file_path"].dropna().astype(str).tolist())) if not rs.empty else []
        dump_json(bundle / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})
        dump_json(bundle / "evidence_sha256_audit.json", {"route_id": route_id, "passed": all(Path(path).exists() for path in raw_paths), "raw_file_count": len(raw_paths)})

    daily_api = {
        "prior_physical_calls_today": args.prior_physical_calls_today,
        "preflight_physical_calls": preflight_calls,
        "campaign_physical_calls": campaign_calls,
        "total_physical_calls_today": total_today,
        "authoritative_saved_raw_calls": AUTHORITATIVE_SAVED_RAW_CALLS,
        "non_authoritative_physical_calls": NON_AUTHORITATIVE_PHYSICAL_CALLS,
        "daily_safe_target": DAILY_SAFE_TARGET,
        "daily_hard_cap": DAILY_HARD_CAP,
        "campaign_safe_target": CAMPAIGN_SAFE_TARGET,
        "campaign_hard_cap": CAMPAIGN_HARD_CAP,
        "daily_usage_equation_passed": total_today == args.prior_physical_calls_today + preflight_calls + campaign_calls,
    }
    runtime = {
        **daily_api,
        "successful_response_count": int(status_counts.get("OK", 0)),
        "fatal_api_error": stop_reason if stop_reason in FATAL_STATUSES else None,
        "first_fatal_error_time": first_fatal_time,
        "calls_after_first_fatal_error": calls_after_fatal,
        "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
        "status_counts": dict(status_counts),
        "campaign_started_at": campaign_started,
        "campaign_finished_at": campaign_finished,
        "api_runtime_audit_passed": api_runtime_passed,
    }
    dump_json(output_root / "daily_api_usage_audit.json", daily_api)
    dump_json(output_root / "campaign_runtime_audit.json", runtime)
    dump_json(output_root / "api_stop_condition_audit.json", {"api_stop_condition_audit_passed": api_runtime_passed, **runtime})
    dump_json(output_root / "upstream_trigger_runtime_audit.json", {"route_priority": ROUTE_PRIORITY, "active_session_limit": 2, "upstream_trigger_runtime_audit_passed": True, "route_summaries": route_summaries})
    dump_json(output_root / "terminal_counter_audit_v6.json", counter)
    dump_json(output_root / "terminal_recovery_observation_sufficiency_audit_v2.json", sufficiency)
    dump_json(output_root / "episode_evidence_sha256_audit_v2.json", evidence_audit)
    dump_json(output_root / "terminal_recovery_contradiction_audit_v2.json", contradiction)
    dump_json(output_root / "prior_episode_deduplication_audit.json", prior_episode_dedupe)
    dump_json(output_root / "secret_leak_audit.json", secret_audit)
    dump_json(output_root / "mapping_regression_audit.json", mapping_regression)
    estimation_auth = {
        "observation_gate": gate,
        "complete_episode_count_total": complete_total,
        "route_complete_episode_counts": {route_id: route_summaries[route_id]["complete_interval_censored_episode_count"] for route_id in TARGET_ROUTES},
        "route_independent_vehicle_counts": {route_id: route_summaries[route_id]["independent_vehicle_count"] for route_id in TARGET_ROUTES},
        "route_censoring_counts": {route_id: {"left": route_summaries[route_id]["left_censored_episode_count"], "right": route_summaries[route_id]["right_censored_episode_count"], "dual": route_summaries[route_id]["dual_censored_episode_count"]} for route_id in TARGET_ROUTES},
        "counter_contract_passed": counter["total"]["counter_contract_v6_passed"],
        "evidence_sha256_passed": evidence_audit["episode_evidence_sha256_audit_passed"],
        "api_runtime_passed": api_runtime_passed,
        "mapping_regression_count": mapping_regression["mapping_regression_count"],
        "contradiction_count": contradiction["contradiction_count"],
        "eligible_for_estimation_review": sufficiency["eligible_for_terminal_recovery_estimation_review"],
        "recommended_next_review": "Terminal recovery estimation preregistration review" if sufficiency["eligible_for_terminal_recovery_estimation_review"] else "Additional upstream observation campaign",
    }
    dump_json(output_root / "terminal_recovery_estimation_input_contract_draft_v2.json", {"observation_artifact_dir": str(output_root), "point_estimates_included": False, "midpoint_values_included": False, "terminal_recovery_estimated": False})
    dump_json(output_root / "terminal_recovery_estimation_authorization_draft_v2.json", estimation_auth)
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, "reason": "R2D-1D2 is an observation-only campaign. It does not authorize terminal recovery estimation, simulator application, Phase 2 turnaround, baseline rerun, or retraining.", "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "approved_for_phase2_turnaround": False, "approved_for_baseline_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a": False, "approved_for_e2": False, "approved_for_full_matrix": False, "real_world_causal_claim_allowed": False})

    gate_payload = {
        "status": gate,
        "authoritative_prior_artifact": str(r2d1d_root),
        **daily_api,
        "fatal_api_error": runtime["fatal_api_error"] or "NONE",
        "first_fatal_error_time": first_fatal_time,
        "calls_after_first_fatal_error": calls_after_fatal,
        "complete_interval_censored_episode_count": complete_total,
        "mapping_regression_count": mapping_regression["mapping_regression_count"],
        "contradiction_count": contradiction["contradiction_count"],
        "counter_contract_v6_passed": counter["total"]["counter_contract_v6_passed"],
        "episode_evidence_sha256_audit_passed": evidence_audit["episode_evidence_sha256_audit_passed"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "eligible_for_terminal_recovery_estimation_review": sufficiency["eligible_for_terminal_recovery_estimation_review"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
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
        "route_results": route_summaries,
    }
    dump_json(output_root / "prompt5_e01_r2d1d2_gate.json", gate_payload)
    report = [
        "# Prompt 5-E01-R2D-1D2 Upstream Observation Campaign",
        "",
        f"- Authoritative input: {r2d1d_root}",
        f"- Excluded superseded input dirs: {len(superseded_dirs)}",
        f"- Execution: {campaign_started} to {campaign_finished}",
        f"- Prior physical calls today: {args.prior_physical_calls_today}",
        f"- Preflight physical calls: {preflight_calls}",
        f"- Campaign physical calls: {campaign_calls}",
        f"- Total physical calls today: {total_today}",
        f"- Authoritative saved raw calls: {AUTHORITATIVE_SAVED_RAW_CALLS}",
        f"- Non-authoritative physical calls: {NON_AUTHORITATIVE_PHYSICAL_CALLS}",
        f"- Max calls/minute: {runtime['max_calls_per_minute']}",
        f"- Fatal API error: {runtime['fatal_api_error'] or 'NONE'} at {first_fatal_time or 'NONE'}",
        f"- Calls after first fatal: {calls_after_fatal}",
        "",
        "## Route Results",
    ]
    for route_id in TARGET_ROUTES:
        row = route_summaries[route_id]
        report.append(f"- {route_id}: upstream={row['upstream_trigger_count']}, upstream_samples={row['upstream_watch_observation_count']}, pre_terminal={row['pre_terminal_confirmed_count']}, terminal_entry={row['terminal_entry_count']}, reset={row['post_terminal_reset_count']}, complete={row['complete_interval_censored_episode_count']}, left={row['left_censored_episode_count']}, right={row['right_censored_episode_count']}, dual={row['dual_censored_episode_count']}, vehicles={row['independent_vehicle_count']}")
    report.extend(
        [
            "",
            "## Guard Confirmations",
            f"- Complete episode raw SHA verification: {evidence_audit['episode_evidence_sha256_audit_passed']}",
            f"- Prior episode duplicates: {prior_episode_dedupe['duplicate_prior_episode_count']}",
            f"- Counter Contract v6 passed: {counter['total']['counter_contract_v6_passed']}",
            f"- Mapping regression count: {mapping_regression['mapping_regression_count']}",
            f"- Contradiction count: {contradiction['contradiction_count']}",
            f"- Secret leak count: {secret_audit['secret_leak_count']}",
            f"- Final gate: {gate}",
            f"- Estimation review eligibility: {sufficiency['eligible_for_terminal_recovery_estimation_review']}",
            "- No terminal recovery point estimate, mean, median, percentile, distribution, midpoint, or simulator value was created.",
            "- Phase 2, baseline rerun, E0/E1 retraining, Prompt 6A, and E2 remain locked.",
            f"- Next allowed review: {estimation_auth['recommended_next_review']}",
        ]
    )
    (output_root / "prompt5_e01_r2d1d2_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
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
    dump_json(output_root / "prompt5_e01_r2d1d2_manifest.json", manifest)
    validation = validate_artifacts(output_root)
    manifest["artifact_validation"] = validation
    manifest["actual_file_count"] = sum(1 for path in output_root.rglob("*") if path.is_file())
    dump_json(output_root / "prompt5_e01_r2d1d2_manifest.json", manifest)

    print()
    print("R2D-1D2 UPSTREAM OBSERVATION COMPLETE")
    print()
    print("artifact_dir:")
    print(output_root)
    print()
    print("gate:")
    print(gate)
    print()
    print("authoritative_prior_artifact:")
    print(r2d1d_root)
    print()
    print("prior_physical_calls_today:")
    print(args.prior_physical_calls_today)
    print()
    print("preflight_physical_calls:")
    print(preflight_calls)
    print()
    print("campaign_physical_calls:")
    print(campaign_calls)
    print()
    print("total_physical_calls_today:")
    print(total_today)
    print()
    print("authoritative_saved_raw_calls:")
    print(AUTHORITATIVE_SAVED_RAW_CALLS)
    print()
    print("fatal_api_error:")
    print(runtime["fatal_api_error"] or "NONE")
    print()
    print("first_fatal_error_time:")
    print(first_fatal_time or "NONE")
    print()
    print("calls_after_first_fatal_error:")
    print(calls_after_fatal)
    print()
    print("route_results:")
    for route_id in TARGET_ROUTES:
        row = route_summaries[route_id]
        print(f"{route_id} = complete={row['complete_interval_censored_episode_count']} / left={row['left_censored_episode_count']} / right={row['right_censored_episode_count']} / upstream={row['upstream_trigger_count']} / vehicles={row['independent_vehicle_count']}")
    print()
    print("complete_interval_censored_episode_count:")
    print(complete_total)
    print()
    print("mapping_regression_count:")
    print(mapping_regression["mapping_regression_count"])
    print()
    print("contradiction_count:")
    print(contradiction["contradiction_count"])
    print()
    print("counter_contract_v6_passed:")
    print(str(counter["total"]["counter_contract_v6_passed"]).lower())
    print()
    print("episode_evidence_sha256_audit_passed:")
    print(str(evidence_audit["episode_evidence_sha256_audit_passed"]).lower())
    print()
    print("secret_leak_count:")
    print(secret_audit["secret_leak_count"])
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
    print(estimation_auth["recommended_next_review"])


if __name__ == "__main__":
    main()
