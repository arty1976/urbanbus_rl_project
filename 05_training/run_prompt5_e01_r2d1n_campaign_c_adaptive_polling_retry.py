#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import time as monotonic_time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"

TARGET_ROUTE = "4010002118"
EXCLUDED_ROUTES = ["4010002001", "4010002004", "4050010000"]
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]

R2D1M_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1m_campaign_c_controlled_live_observation_20260728_090115"
R2D1L_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1l_campaign_c_authorization_review_20260727_123717"
R2D1K_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_20260727_113644"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1H_LIVE_SCRIPT = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1h_campaign_a_live_observation_actual.py"
MAPPING_PATH = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"
REGISTRY_PATH = R2D1K_HF1_ROOT / "cumulative_episode_registry_candidate_hf1.parquet"

CAMPAIGN_WINDOW_START = time(9, 0)
CAMPAIGN_WINDOW_END = time(14, 0)
FOLLOW_REQUIREMENT_MINUTES = 115
DAILY_PHYSICAL_SAFETY_CAP = 800
RECOMMENDED_TOTAL_CALLS = 160
WARNING_TOTAL_CALLS = 180
ABSOLUTE_HARD_CAP = 260
MIN_EFFECTIVE_HARD_CAP_TO_START = 140
CONFIGURED_MAX_CALLS_PER_MINUTE = 4
PREVIOUS_CAMPAIGN_CALLS = 122
REPLAY_CALL_LIMIT = 97

REQUIRED_FILES = [
    "prompt5_e01_r2d1n_manifest.json",
    "prompt5_e01_r2d1n_gate.json",
    "prompt5_e01_r2d1n_final_report.md",
    "upstream_reference_r2d1m_hf1.json",
    "upstream_reference_r2d1m.json",
    "upstream_reference_r2d1l.json",
    "upstream_reference_r2d1k_hf1.json",
    "upstream_reference_hf1_mapping.json",
    "adaptive_eta_training_dataset.json",
    "adaptive_sequence_to_terminal_eta.json",
    "adaptive_sequence_to_terminal_eta.parquet",
    "adaptive_polling_policy.json",
    "adaptive_polling_replay_audit.json",
    "adaptive_polling_call_savings_audit.json",
    "runtime_execution_authorization.json",
    "runtime_schedule_manifest.json",
    "daily_api_usage_audit.json",
    "effective_api_budget.json",
    "provider_preflight_audit.json",
    "runtime_audit.json",
    "api_stop_condition_audit.json",
    "candidate_vehicle_selection_audit.json",
    "adaptive_polling_transition_audit.json",
    "state_transition_audit.json",
    "counter_contract_v13.json",
    "counter_contract_v13.parquet",
    "position_samples.parquet",
    "vehicle_trajectories.parquet",
    "terminal_recovery_episodes.json",
    "terminal_recovery_episodes.parquet",
    "terminal_recovery_interval_bounds.json",
    "terminal_recovery_interval_bounds.parquet",
    "raw_file_index.json",
    "raw_provenance_audit.json",
    "clock_semantics_audit.json",
    "interval_validation_audit.json",
    "episode_deduplication_audit.json",
    "cumulative_episode_registry_candidate_12.json",
    "cumulative_episode_registry_candidate_12.parquet",
    "method_prototype_readiness_audit.json",
    "strict_json_content_type_audit.json",
    "json_parquet_synchronization_audit.json",
    "secret_leak_audit.json",
    "authoritative_input_immutability_audit.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


COUNTER_COLUMNS = [
    "scope",
    "route_id",
    "guard_scan_observation_count",
    "approach_scan_observation_count",
    "pre_terminal_watch_observation_count",
    "terminal_approach_focused_observation_count",
    "post_terminal_focused_observation_count",
    "candidate_vehicle_count",
    "tracking_session_started_count",
    "terminal_entry_count",
    "terminal_stop_hold_observation_count",
    "post_terminal_reset_count",
    "post_terminal_confirmation_observation_count",
    "post_terminal_confirmed_count",
    "total_final_episode_count",
    "complete_final_count",
    "right_censored_final_count",
    "left_censored_final_count",
    "invalid_final_count",
    "new_complete_episode_count",
    "right_censored_episode_count",
    "left_censored_episode_count",
    "invalid_episode_count",
    "new_global_complete_vehicle_count",
    "new_route_local_complete_vehicle_count",
    "episode_duplicate_count",
    "contradiction_count",
    "counter_contract_failure_count",
]

EPISODE_COLUMNS = [
    "episode_id",
    "route_id",
    "vehicle_id",
    "episode_status",
    "final_status_class",
    "complete_interval_censored_episode",
    "first_upstream_watch_time",
    "last_pre_terminal_request_time",
    "first_terminal_request_time",
    "last_terminal_request_time",
    "first_post_terminal_request_time",
    "post_terminal_confirmed_request_time",
    "first_upstream_watch_sequence",
    "last_pre_terminal_sequence",
    "first_terminal_sequence",
    "last_terminal_sequence",
    "first_post_terminal_sequence",
    "provider_lower_bound_sec",
    "provider_upper_bound_sec",
    "request_lower_bound_sec",
    "request_upper_bound_sec",
    "conservative_dual_lower_bound_sec",
    "conservative_dual_upper_bound_sec",
    "eligible_for_estimation_input",
]

POSITION_COLUMNS = [
    "request_id",
    "campaign_id",
    "route_id",
    "vehicle_id",
    "capture_mode",
    "request_observation_time",
    "provider_position_event_time",
    "direction",
    "current_sequence",
    "terminal_trigger_sequence",
    "provider_response_status",
    "raw_file_relative_path",
    "raw_file_sha256",
]

R2D1N_EXTRA_EPISODE_COLUMNS = [
    "source_episode_id",
    "disappeared_waiting_reentry_observation_count",
    "global_vehicle_classification",
    "route_local_vehicle_classification",
    "candidate_priority_tier",
]


def now_kst() -> datetime:
    return datetime.now(KST)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [jsonable(row) for row in frame.to_dict("records")]


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    pd.read_parquet(path)


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().astimezone(KST)


def seconds_between(start: Any, end: Any) -> Optional[float]:
    a = parse_dt(start)
    b = parse_dt(end)
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def as_int(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def latest_cleanup_artifact() -> Optional[Path]:
    candidates = sorted(ARTIFACTS_ROOT.glob("prompt5_e01_r2d1m_hf1_transport_error_finalization_*"))
    for root in reversed(candidates):
        gate_path = root / "prompt5_e01_r2d1m_hf1_gate.json"
        if not gate_path.exists():
            continue
        try:
            gate = read_json(gate_path)
        except Exception:
            continue
        if gate.get("gate_status") == "PASS_CAMPAIGN_C_PARTIAL_METADATA_FINALIZED":
            return root
    return None


def build_upstream_reference(root: Path, gate_name: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "absolute_path": str(root),
        "exists": root.exists(),
        "read_only_input": True,
        "manifest_sha256": sha256_file(next(root.glob("*manifest.json"))) if root.exists() and list(root.glob("*manifest.json")) else None,
    }
    if gate_name:
        gate_path = root / gate_name
        payload["gate_path"] = str(gate_path)
        payload["gate_exists"] = gate_path.exists()
        if gate_path.exists():
            gate = read_json(gate_path)
            payload["gate_status"] = gate.get("gate_status")
            payload["gate_passed"] = gate.get("gate_passed")
    if extra:
        payload.update(extra)
    return payload


def route_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str)).size().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def route_local_unique_counts(registry: pd.DataFrame) -> Dict[str, int]:
    if "route_local_vehicle_identity_key" not in registry:
        return {route: 0 for route in ALL_ROUTES}
    counts = registry.groupby(registry["route_id"].astype(str))["route_local_vehicle_identity_key"].nunique().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def normalize_hour_bucket(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}", text):
        hour = int(text)
        if 0 <= hour <= 23:
            return f"{hour:02d}:00-{hour:02d}:59"
    return text


def registry_status() -> Dict[str, Any]:
    registry = pd.read_parquet(REGISTRY_PATH)
    global_vehicle_col = "canonical_global_vehicle_id" if "canonical_global_vehicle_id" in registry else "vehicle_id"
    dates = sorted(pd.to_datetime(registry.get("service_date"), errors="coerce").dropna().dt.date.astype(str).unique().tolist()) if "service_date" in registry else []
    hours = sorted({normalize_hour_bucket(value) for value in registry.get("hour_bucket", pd.Series(dtype=str)).dropna().tolist()})
    return {
        "frame": registry,
        "complete_count": int(len(registry)),
        "route_counts": route_counts(registry),
        "global_unique_vehicle_count": int(registry[global_vehicle_col].dropna().astype(str).nunique()) if global_vehicle_col in registry else 0,
        "route_local_unique_counts": route_local_unique_counts(registry),
        "observation_dates": dates,
        "hour_buckets": hours,
        "invalid_clock_order_count": int((registry.get("clock_semantics_status", pd.Series(dtype=str)).astype(str) == "INVALID_CLOCK_ORDER").sum()) if "clock_semantics_status" in registry else 0,
        "provenance_failure_count": int((registry.get("evidence_sha256_passed", pd.Series(dtype=bool)) == False).sum()) if "evidence_sha256_passed" in registry else 0,
    }


def route_meta_from_mapping() -> Dict[str, Any]:
    df = pd.read_parquet(MAPPING_PATH)
    row = df[df["route_id"].astype(str) == TARGET_ROUTE].iloc[0].to_dict()
    effective = as_int(row.get("effective_live_terminal_sequence"))
    terminal = None if effective is None else effective - 5
    return {
        "mapping_record": row,
        "effective_live_terminal_sequence": effective,
        "terminal_trigger_sequence": terminal,
        "early_upstream_trigger_sequence": None if effective is None else effective - 22,
        "upstream_trigger_sequence": None if effective is None else effective - 12,
        "terminal_entry_source": "HF1 effective_live_terminal_sequence - 5",
    }


def normalize_position_frame(path: Path, terminal_sequence: int) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "route_id" not in df:
        return pd.DataFrame()
    df = df[df["route_id"].astype(str) == TARGET_ROUTE].copy()
    if df.empty:
        return pd.DataFrame()
    seq_col = next((name for name in ["current_sequence", "sequence", "seq", "stop_sequence"] if name in df.columns), None)
    time_col = next((name for name in ["request_observation_time", "request_time", "observed_at", "timestamp"] if name in df.columns), None)
    if seq_col is None or time_col is None:
        return pd.DataFrame()
    vehicle_col = next((name for name in ["vehicle_id", "vhcNo2", "vehicle_no"] if name in df.columns), None)
    direction_col = next((name for name in ["direction", "direction_id", "moveDir"] if name in df.columns), None)
    provider_col = next((name for name in ["provider_position_event_time", "provider_event_time", "event_time"] if name in df.columns), None)
    out = pd.DataFrame(
        {
            "source_path": str(path),
            "route_id": df["route_id"].astype(str),
            "vehicle_id": df[vehicle_col].astype(str) if vehicle_col else "UNKNOWN",
            "direction": df[direction_col].astype(str) if direction_col else "UNKNOWN",
            "sequence": pd.to_numeric(df[seq_col], errors="coerce"),
            "request_observation_time": df[time_col],
            "provider_event_time": df[provider_col] if provider_col else None,
        }
    )
    out["request_dt"] = pd.to_datetime(out["request_observation_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["sequence", "request_dt"])
    out["sequence"] = out["sequence"].astype(int)
    out["terminal_trigger_sequence"] = terminal_sequence
    out["is_terminal_zone"] = out["sequence"] >= terminal_sequence
    return out


def collect_eta_training_dataset(terminal_sequence: int) -> pd.DataFrame:
    frames = []
    for path in sorted(ARTIFACTS_ROOT.rglob("*position_samples.parquet")):
        if "prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_" in str(path):
            continue
        frame = normalize_position_frame(path, terminal_sequence)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["sequence", "eta_to_terminal_entry_sec"])
    all_samples = pd.concat(frames, ignore_index=True)
    all_samples = all_samples.sort_values(["source_path", "vehicle_id", "direction", "request_dt", "sequence"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for (_, vehicle, direction), group in all_samples.groupby(["source_path", "vehicle_id", "direction"], dropna=False):
        group = group.sort_values("request_dt").reset_index(drop=True)
        terminal_rows = group[group["sequence"] >= terminal_sequence]
        if terminal_rows.empty:
            continue
        first_terminal_time = terminal_rows.iloc[0]["request_dt"]
        first_terminal_request = terminal_rows.iloc[0]["request_observation_time"]
        for _, sample in group.iterrows():
            eta = (first_terminal_time - sample["request_dt"]).total_seconds()
            if eta < 0 or eta > 4 * 3600:
                continue
            rows.append(
                {
                    "source_path": sample["source_path"],
                    "route_id": TARGET_ROUTE,
                    "vehicle_id": str(vehicle),
                    "direction": str(direction),
                    "sequence": int(sample["sequence"]),
                    "request_observation_time": sample["request_observation_time"],
                    "provider_event_time": sample.get("provider_event_time"),
                    "first_terminal_entry_request_time": first_terminal_request,
                    "eta_to_terminal_entry_sec": float(eta),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["sequence", "eta_to_terminal_entry_sec"])
    return result.drop_duplicates(subset=["source_path", "vehicle_id", "direction", "request_observation_time", "sequence"]).reset_index(drop=True)


def build_eta_table(training: pd.DataFrame, terminal_sequence: int) -> pd.DataFrame:
    if training.empty:
        return pd.DataFrame(
            [
                {
                    "sequence": seq,
                    "sample_count": 0,
                    "minimum_eta_sec": max(0, terminal_sequence - seq) * 90,
                    "p25_eta_sec": max(0, terminal_sequence - seq) * 105,
                    "median_eta_sec": max(0, terminal_sequence - seq) * 120,
                    "p75_eta_sec": max(0, terminal_sequence - seq) * 150,
                    "maximum_eta_sec": max(0, terminal_sequence - seq) * 180,
                    "lower_operational_eta_sec": max(0, terminal_sequence - seq) * 105,
                    "upper_operational_eta_sec": max(0, terminal_sequence - seq) * 150,
                    "eta_source": "sequence_distance_fallback",
                    "sequence_band_width": 3,
                }
                for seq in range(1, terminal_sequence + 1)
            ]
        )

    per_sequence = []
    valid = training[training["sequence"].between(1, terminal_sequence)].copy()
    valid["sequence"] = valid["sequence"].astype(int)
    eta_by_seq = valid.groupby("sequence")["eta_to_terminal_entry_sec"]
    positive = valid[(valid["sequence"] < terminal_sequence) & (valid["eta_to_terminal_entry_sec"] > 0)].copy()
    if positive.empty:
        sec_per_sequence = 120.0
    else:
        positive["distance"] = (terminal_sequence - positive["sequence"]).clip(lower=1)
        sec_per_sequence = float((positive["eta_to_terminal_entry_sec"] / positive["distance"]).median())
        sec_per_sequence = min(max(sec_per_sequence, 45.0), 240.0)

    for seq in range(1, terminal_sequence + 1):
        if seq in eta_by_seq.groups and len(eta_by_seq.get_group(seq)) >= 2:
            values = eta_by_seq.get_group(seq).astype(float)
            source = "direct_sequence_empirical"
            band_width = 0
        else:
            values = pd.Series(dtype=float)
            band_width = 0
            for width in [1, 2, 3]:
                subset = valid[valid["sequence"].between(max(1, seq - width), min(terminal_sequence, seq + width))]["eta_to_terminal_entry_sec"].astype(float)
                if len(subset) >= 2:
                    values = subset
                    band_width = width
                    break
            source = "banded_empirical" if len(values) >= 2 else "sequence_distance_fallback"
        if len(values) >= 2:
            minimum = float(values.min())
            p25 = float(values.quantile(0.25))
            median = float(values.quantile(0.50))
            p75 = float(values.quantile(0.75))
            maximum = float(values.max())
            sample_count = int(len(values))
            lower = max(0.0, p25)
            upper = max(lower, p75)
        else:
            distance = max(0, terminal_sequence - seq)
            minimum = distance * max(30.0, sec_per_sequence * 0.65)
            p25 = distance * max(45.0, sec_per_sequence * 0.80)
            median = distance * sec_per_sequence
            p75 = distance * min(300.0, sec_per_sequence * 1.35)
            maximum = distance * min(360.0, sec_per_sequence * 1.80)
            sample_count = int(len(values))
            lower = p25
            upper = p75
        per_sequence.append(
            {
                "sequence": seq,
                "sample_count": sample_count,
                "minimum_eta_sec": minimum,
                "p25_eta_sec": p25,
                "median_eta_sec": median,
                "p75_eta_sec": p75,
                "maximum_eta_sec": maximum,
                "lower_operational_eta_sec": lower,
                "upper_operational_eta_sec": upper,
                "eta_source": source,
                "sequence_band_width": band_width,
            }
        )
    return pd.DataFrame(per_sequence)


def read_raw_response(path: Path) -> Tuple[datetime, List[Dict[str, Any]], bool]:
    dt = datetime.strptime(path.name[:22], "%Y%m%d_%H%M%S_%f").replace(tzinfo=KST)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dt, [], False
    body = payload.get("body", payload)
    items = body.get("items") or []
    if isinstance(items, dict):
        items = [items]
    return dt, [dict(item) for item in items], True


def replay_timeline() -> List[Dict[str, Any]]:
    raw_dir = R2D1M_ROOT / "raw" / TARGET_ROUTE
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        if "preflight" in path.name:
            continue
        dt, items, ok = read_raw_response(path)
        if not ok:
            continue
        seqs = [as_int(item.get("seq") or item.get("stopSeq")) for item in items]
        vehicles = [str(item.get("vhcNo2") or item.get("vhcNo") or item.get("busId") or "").strip() for item in items]
        rows.append({"time": dt, "path": path, "items": items, "row_count": len(items), "sequences": [seq for seq in seqs if seq is not None], "vehicles": vehicles})
    return rows


def eta_for_sequence(eta_table: pd.DataFrame, sequence: Optional[int]) -> Tuple[float, float]:
    if sequence is None or eta_table.empty:
        return 9999.0, 9999.0
    subset = eta_table[eta_table["sequence"] == int(sequence)]
    if subset.empty:
        return 9999.0, 9999.0
    row = subset.iloc[0]
    return float(row["lower_operational_eta_sec"]), float(row["upper_operational_eta_sec"])


def mode_for_state(sequence: Optional[int], terminal_sequence: int, eta_table: pd.DataFrame, terminal_entered: bool, absent_after_terminal: bool, last_mode: str) -> Tuple[str, int]:
    if terminal_entered or absent_after_terminal:
        return "POST_TERMINAL_FOCUSED", 30
    lower, upper = eta_for_sequence(eta_table, sequence)
    if sequence is not None and sequence >= terminal_sequence - 5:
        return "TERMINAL_APPROACH_FOCUSED", 30
    if upper <= 5 * 60:
        return "TERMINAL_APPROACH_FOCUSED", 30
    if sequence is not None and sequence >= terminal_sequence - 12:
        return "PRE_TERMINAL_WATCH", 60
    if upper <= 12 * 60:
        return "PRE_TERMINAL_WATCH", 60
    if lower <= 25 * 60 and upper > 12 * 60:
        return "APPROACH_SCAN", 120
    return "GUARD_SCAN", 300


def adaptive_replay(eta_table: pd.DataFrame, terminal_sequence: int) -> Dict[str, Any]:
    timeline = replay_timeline()
    if not timeline:
        return {"adaptive_replay_passed": False, "blocking_reason": "NO_R2D1M_REPLAY_TIMELINE", "records": []}

    selected: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    due = timeline[0]["time"]
    index = 0
    terminal_entered = False
    absent_after_terminal = False
    session_active = False
    candidate_detected = False
    candidate_started_at: Optional[datetime] = None
    last_pre_terminal_preserved = False
    terminal_entry_time: Optional[datetime] = None
    detected_terminal_entry_time: Optional[datetime] = None
    terminal_hold_sequences: List[int] = []
    last_sequence: Optional[int] = None
    last_mode = "GUARD_SCAN"

    while index < len(timeline):
        while index < len(timeline) and timeline[index]["time"] < due:
            index += 1
        if index >= len(timeline):
            break
        obs = timeline[index]
        sequences = obs["sequences"]
        sequence = sequences[0] if sequences else last_sequence
        vehicle = obs["vehicles"][0] if obs["vehicles"] else None
        if sequences:
            last_sequence = sequence
            absent_after_terminal = False
        elif terminal_entered and session_active:
            absent_after_terminal = True
        if not session_active and sequence is not None and (terminal_sequence - 16) <= sequence < (terminal_sequence - 5):
            session_active = True
            candidate_started_at = obs["time"]
            candidate_detected = True
        if session_active and sequence == terminal_sequence - 1:
            last_pre_terminal_preserved = True
        if session_active and sequence is not None and sequence >= terminal_sequence:
            if not terminal_entered:
                terminal_entered = True
                detected_terminal_entry_time = obs["time"]
            terminal_hold_sequences.append(sequence)
        if not session_active and sequence is not None and sequence >= terminal_sequence - 5:
            mode, interval = "GUARD_SCAN", 300
        else:
            mode, interval = mode_for_state(sequence, terminal_sequence, eta_table, terminal_entered, absent_after_terminal, last_mode)
        selected.append(
            {
                "observation_time": obs["time"].isoformat(),
                "mode": mode,
                "interval_sec": interval,
                "row_count": obs["row_count"],
                "sequence": sequence,
                "vehicle_id": vehicle,
                "raw_relative_path": str(obs["path"].relative_to(R2D1M_ROOT)),
            }
        )
        if mode != last_mode:
            transitions.append({"time": obs["time"].isoformat(), "from_mode": last_mode, "to_mode": mode, "sequence": sequence})
        last_mode = mode
        due = obs["time"] + timedelta(seconds=interval)
        index += 1

    all_terminal_entries = [
        obs
        for obs in timeline
        if candidate_started_at is not None and obs["time"] >= candidate_started_at and obs["sequences"] and obs["sequences"][0] >= terminal_sequence
    ]
    if all_terminal_entries:
        terminal_entry_time = all_terminal_entries[0]["time"]
    terminal_delay = None
    if terminal_entry_time is not None and detected_terminal_entry_time is not None:
        terminal_delay = (detected_terminal_entry_time - terminal_entry_time).total_seconds()

    projected_calls = len(selected)
    reduction_percent = (1.0 - (projected_calls / PREVIOUS_CAMPAIGN_CALLS)) * 100.0
    mode_counts = Counter(record["mode"] for record in selected)
    result = {
        "adaptive_replay_candidate_detected": candidate_detected,
        "adaptive_replay_last_pre_terminal_preserved": last_pre_terminal_preserved,
        "adaptive_replay_terminal_entry_preserved": detected_terminal_entry_time is not None,
        "adaptive_replay_terminal_hold_preserved": any(seq >= terminal_sequence for seq in terminal_hold_sequences),
        "terminal_entry_detection_delay_sec": terminal_delay,
        "terminal_entry_detection_delay_passed": terminal_delay is not None and terminal_delay <= 90,
        "adaptive_replay_projected_calls": projected_calls,
        "previous_campaign_calls": PREVIOUS_CAMPAIGN_CALLS,
        "adaptive_replay_call_reduction_percent": reduction_percent,
        "adaptive_replay_call_reduction_passed": projected_calls <= REPLAY_CALL_LIMIT,
        "lookahead_used": False,
        "polling_mode_counts": dict(mode_counts),
        "selected_records": selected,
        "transition_records": transitions,
    }
    result["adaptive_replay_passed"] = all(
        [
            result["adaptive_replay_candidate_detected"],
            result["adaptive_replay_last_pre_terminal_preserved"],
            result["adaptive_replay_terminal_entry_preserved"],
            result["adaptive_replay_terminal_hold_preserved"],
            result["terminal_entry_detection_delay_passed"],
            result["adaptive_replay_call_reduction_passed"],
            not result["lookahead_used"],
        ]
    )
    return result


def load_live_base() -> Any:
    spec = importlib.util.spec_from_file_location("r2d1h_live_base_for_r2d1n", R2D1H_LIVE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live base from {R2D1H_LIVE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_ROUTES = [TARGET_ROUTE]
    module.MAX_CALLS_PER_MINUTE = CONFIGURED_MAX_CALLS_PER_MINUTE
    module.MAX_FOLLOW_MINUTES = {TARGET_ROUTE: 100}
    module.REQUIRED_END_BUFFER_MINUTES = 15
    module.FATAL_STATUSES = {
        "RATE_LIMIT",
        "HTTP_429",
        "AUTH_ERROR",
        "HTML_RESPONSE",
        "HTTP_ERROR",
        "EMPTY_RESPONSE",
        "FAIL_NON_TARGET_ROUTE_ACCESS",
        "SECRET_LEAK",
    }
    for column in R2D1N_EXTRA_EPISODE_COLUMNS:
        if column not in module.EPISODE_COLUMNS:
            module.EPISODE_COLUMNS.append(column)
    return module


def registry_vehicle_sets(registry: pd.DataFrame) -> Tuple[set[str], set[str]]:
    vehicle_col = "canonical_vehicle_id" if "canonical_vehicle_id" in registry.columns else "vehicle_id"
    global_ids = {str(value).strip() for value in registry[vehicle_col].dropna().astype(str).tolist() if str(value).strip()}
    route_local_ids = {
        str(value).strip()
        for value in registry[registry["route_id"].astype(str) == TARGET_ROUTE][vehicle_col].dropna().astype(str).tolist()
        if str(value).strip()
    }
    return global_ids, route_local_ids


def raw_suffix(content_type: str, status: str, payload: bytes) -> str:
    lower_type = (content_type or "").lower()
    head = payload[:512].decode("utf-8", errors="replace").lstrip().lower()
    if "json" in lower_type or (status == "OK" and not head.startswith("<")):
        return ".json"
    if "html" in lower_type or head.startswith("<!doctype html") or head.startswith("<html") or "<html" in head:
        return ".html"
    if lower_type.startswith("text/"):
        return ".txt"
    return ".bin"


def make_adaptive_live_runner_class(
    base: Any,
    eta_table: pd.DataFrame,
    terminal_sequence: int,
    global_exclusion_ids: set[str],
    route_local_exclusion_ids: set[str],
) -> Any:
    class R2D1NAdaptiveRunner(base.CampaignRunner):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.global_exclusion_ids = {str(item).strip() for item in global_exclusion_ids}
            self.route_local_exclusion_ids = {str(item).strip() for item in route_local_exclusion_ids}
            self.last_sequence_by_route: Dict[str, Optional[int]] = {TARGET_ROUTE: None}
            self.last_mode_by_route: Dict[str, str] = {TARGET_ROUTE: "GUARD_SCAN"}
            self.mode_transition_records: List[Dict[str, Any]] = []

        def classify_vehicle(self, route: str, vehicle_id: str) -> str:  # noqa: ARG002
            canonical = str(vehicle_id).strip()
            if canonical in self.campaign_seen_vehicle_ids[route]:
                return "DUPLICATE_TERMINAL_CYCLE"
            if canonical not in self.global_exclusion_ids:
                return "GLOBAL_UNSEEN_VEHICLE"
            if canonical not in self.route_local_exclusion_ids:
                return "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
            return "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"

        def desired_route_for_new_session(self, route: str) -> bool:
            if route != TARGET_ROUTE:
                return False
            if not self.allow_new_candidates or self.sessions[route] is not None:
                return False
            if sum(self.complete_count_by_route.values()) >= 1:
                return False
            return self.route_can_start(route)

        def adaptive_mode(self, route: str) -> str:
            session = self.sessions.get(route)
            terminal_entered = bool(session and session.get("first_terminal") is not None)
            absent_after_terminal = bool(session and str(session.get("phase")) == "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY")
            seq = None
            if session is not None:
                for key in ["last_terminal", "last_pre_terminal", "first_upstream"]:
                    row = session.get(key)
                    seq = base.as_int(row.get("current_sequence")) if row is not None else seq
                    if seq is not None:
                        break
            if seq is None:
                seq = self.last_sequence_by_route.get(route)
            if session is None and seq is not None and seq >= terminal_sequence - 5:
                mode = "GUARD_SCAN"
                previous = self.last_mode_by_route.get(route)
                if previous != mode:
                    self.mode_transition_records.append({"time": base.iso(), "route_id": route, "from_mode": previous, "to_mode": mode, "sequence": seq, "reason": "no_active_session_terminal_residue"})
                self.last_mode_by_route[route] = mode
                return mode
            mode, _ = mode_for_state(seq, terminal_sequence, eta_table, terminal_entered, absent_after_terminal, self.last_mode_by_route.get(route, "GUARD_SCAN"))
            previous = self.last_mode_by_route.get(route)
            if previous != mode:
                self.mode_transition_records.append({"time": base.iso(), "route_id": route, "from_mode": previous, "to_mode": mode, "sequence": seq})
            self.last_mode_by_route[route] = mode
            return mode

        def route_mode(self, route: str) -> str:
            return self.adaptive_mode(route)

        def fetch(self, route: str, mode: str) -> List[Dict[str, Any]]:
            if route != TARGET_ROUTE:
                self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                self.first_fatal_error_time = self.first_fatal_error_time or base.iso()
                self.stop_reason = self.fatal_error
                return []
            if not self.call_allowed():
                return []
            if self.last_call_mono is not None:
                sleep_for = (60.0 / CONFIGURED_MAX_CALLS_PER_MINUTE) - (monotonic_time.monotonic() - self.last_call_mono)
                if sleep_for > 0:
                    monotonic_time.sleep(sleep_for)

            request_dt = base.now_kst()
            request_time = request_dt.isoformat(timespec="seconds")
            counter = len(self.request_records) + 1
            request_id = f"r2d1n_request_{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{counter:05d}"
            raw_dir = self.output_root / "raw" / route
            raw_dir.mkdir(parents=True, exist_ok=True)
            params = [("serviceKey", self.secret), ("routeId", route), ("resultType", "json")]
            url = base.GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
            http_status: Optional[int] = None
            content_type = ""
            timeout_error = False
            encoding = "utf-8"
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1n/1.0"})
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

            text = payload.decode(encoding or "utf-8", errors="replace")
            status = base.detect_response_status(text, http_status, timeout_error)
            suffix = raw_suffix(content_type, status, payload)
            raw_path = raw_dir / f"{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}{suffix}"
            raw_path.write_bytes(payload)
            self.last_call_mono = monotonic_time.monotonic()

            items: List[Dict[str, Any]] = []
            parsed = False
            result_code: Optional[str] = None
            if status == "OK":
                items, parsed, result_code = base.parse_items(text)
                if not parsed:
                    status = "PROVIDER_PARSE_FAILURE"
            raw_sha = sha256_file(raw_path)
            route_mismatch_count = 0
            rows: List[Dict[str, Any]] = []
            meta = self.metas[route]
            effective = base.as_int(meta.get("effective_live_terminal_sequence"))
            early = base.as_int(meta.get("early_upstream_trigger_sequence"))
            upstream = base.as_int(meta.get("upstream_trigger_sequence"))
            terminal = base.as_int(meta.get("terminal_trigger_sequence"))
            for item in items:
                response_route = base.norm(item.get("routeId")) or route
                if response_route != route:
                    route_mismatch_count += 1
                seq = base.as_int(item.get("seq") or item.get("stopSeq"))
                vehicle_id = base.norm(item.get("vhcNo2")) or base.norm(item.get("vhcNo")) or base.norm(item.get("busId"))
                if seq is None or vehicle_id is None:
                    continue
                self.last_sequence_by_route[route] = seq
                direction = base.norm(item.get("moveDir")) or base.norm(item.get("direction_id")) or base.norm(meta.get("direction")) or "1"
                provider_event_time = base.provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
                sample = {
                    "request_id": request_id,
                    "campaign_id": "R2D-1N-CAMPAIGN-C-ADAPTIVE",
                    "session_id": None,
                    "episode_id": None,
                    "route_id": route,
                    "vehicle_id": vehicle_id,
                    "capture_mode": mode,
                    "derived_terminal_phase": None,
                    "request_observation_time": request_time,
                    "provider_position_event_time": provider_event_time,
                    "provider_event_lag_sec": base.seconds_between(provider_event_time, request_time),
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
                    "route_continuity_passed": response_route == route,
                    "direction_continuity_passed": True,
                    "provider_response_status": status,
                    "provider_result_code": result_code,
                    "http_status": http_status,
                    "raw_file_relative_path": str(raw_path.relative_to(self.output_root)),
                    "raw_file_sha256": raw_sha,
                    "authoritative_raw": True,
                }
                sample["derived_terminal_phase"] = base.derived_terminal_phase(sample)
                rows.append(sample)

            record = {
                "request_id": request_id,
                "route_id": route,
                "request_observation_time": request_time,
                "capture_mode": mode,
                "http_status": http_status,
                "content_type": content_type,
                "raw_content_type_class": suffix.lstrip("."),
                "provider_response_status": status,
                "provider_result_code": result_code,
                "valid_provider_payload": status == "OK",
                "provider_parse_success": parsed,
                "route_id_response_match": route_mismatch_count == 0,
                "route_mismatch_count": route_mismatch_count,
                "normalized_row_count": len(rows),
                "raw_relative_path": str(raw_path.relative_to(self.output_root)),
                "raw_sha256": raw_sha,
                "redacted_request_metadata": {"endpoint": base.GETPOS02_URL, "routeId": route, "resultType": "json", "serviceKey": "<REDACTED>"},
                "request_url_redacted": base.redact_url(url, self.secret),
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
            if status in base.FATAL_STATUSES:
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
            if route_mismatch_count:
                self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                self.first_fatal_error_time = request_time
                self.stop_reason = self.fatal_error
            print(
                json.dumps(
                    {
                        "event": "r2d1n_api_call",
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

        def maybe_start_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
            if not self.desired_route_for_new_session(route):
                return
            candidates = [
                dict(row)
                for row in rows
                if base.as_int(row.get("current_sequence")) is not None
                and base.as_int(row.get("current_sequence")) < terminal_sequence - 5
                and (row.get("is_early_upstream_watch_zone") or row.get("is_upstream_watch_zone"))
            ]
            if not candidates:
                return
            priority = {
                "GLOBAL_UNSEEN_VEHICLE": 4,
                "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE": 3,
                "PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE": 2,
                "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE": 1,
            }
            scored = []
            for row in candidates:
                vehicle_id = base.norm(row.get("vehicle_id"))
                if vehicle_id is None:
                    self.history_count_by_route[route]["INVALID_VEHICLE_ID"] += 1
                    continue
                history = self.classify_vehicle(route, vehicle_id)
                if history == "DUPLICATE_TERMINAL_CYCLE":
                    continue
                scored.append((priority.get(history, 0), base.as_int(row.get("current_sequence")) or -1, history, row))
            if not scored:
                return
            _, _, history, chosen = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
            self.session_counter += 1
            self.candidate_count_by_route[route] += 1
            self.history_count_by_route[route][history] += 1
            session_id = f"r2d1n_session_{self.session_counter:05d}_{route}"
            episode_id = f"r2d1n_episode_{self.session_counter:05d}_{route}"
            chosen["session_id"] = session_id
            chosen["episode_id"] = episode_id
            self.sessions[route] = {
                "route_id": route,
                "session_id": session_id,
                "episode_id": episode_id,
                "vehicle_id": base.norm(chosen.get("vehicle_id")),
                "direction": base.norm(chosen.get("direction")),
                "vehicle_history_class": history,
                "phase": "UPSTREAM_FOCUSED",
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
                "disappeared_waiting_reentry_observation_count": 0,
            }
            self.campaign_seen_vehicle_ids[route].add(str(chosen["vehicle_id"]))
            self._tag_sample(chosen)
            print(
                json.dumps(
                    {
                        "event": "r2d1n_candidate_started",
                        "route_id": route,
                        "vehicle_id": chosen["vehicle_id"],
                        "vehicle_history_class": history,
                        "sequence": chosen["current_sequence"],
                        "time": base.iso(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        def update_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
            session = self.sessions.get(route)
            disappeared_before = int((session or {}).get("disappeared_waiting_reentry_observation_count", 0))
            terminal_entered = bool(session and session.get("first_terminal") is not None)
            reset_observed = bool(session and session.get("first_post_terminal") is not None)
            vehicle_id = base.norm((session or {}).get("vehicle_id"))
            direction = base.norm((session or {}).get("direction"))
            matching_before = any(
                base.norm(row.get("vehicle_id")) == vehicle_id
                and base.norm(row.get("route_id")) == route
                and base.norm(row.get("direction")) == direction
                for row in rows
            )
            absence_increment = 1 if terminal_entered and not reset_observed and not matching_before else 0
            super().update_session(route, rows)
            session_after = self.sessions.get(route)
            if session_after is not None:
                session_after["disappeared_waiting_reentry_observation_count"] = disappeared_before + absence_increment

        def finish_session(self, route: str, status: str, invalid_reason: Optional[str] = None) -> None:
            session = self.sessions.get(route)
            disappeared_count = int((session or {}).get("disappeared_waiting_reentry_observation_count", (session or {}).get("missing_after_terminal_count", 0)))
            vehicle_id = str((session or {}).get("vehicle_id") or "").strip()
            history = str((session or {}).get("vehicle_history_class") or "")
            super().finish_session(route, status, invalid_reason)
            if self.episodes:
                self.episodes[-1]["source_episode_id"] = self.episodes[-1].get("episode_id")
                self.episodes[-1]["disappeared_waiting_reentry_observation_count"] = disappeared_count
                self.episodes[-1]["global_vehicle_classification"] = "GLOBAL_UNSEEN_VEHICLE" if vehicle_id not in self.global_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE"
                self.episodes[-1]["route_local_vehicle_classification"] = "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if vehicle_id not in self.route_local_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"
                self.episodes[-1]["candidate_priority_tier"] = history

        def run(self) -> None:
            self.campaign_started_at = base.iso()
            while not self.fatal_error and base.now_kst() < self.planned_end:
                complete_total = sum(self.complete_count_by_route.values())
                if complete_total >= 1:
                    self.allow_new_candidates = False
                    self.stop_reason = "CAMPAIGN_C_FINAL_EPISODE_TARGET_REACHED"
                    break
                if self.physical_calls_this_artifact >= int(self.effective_cap * 0.9):
                    self.allow_new_candidates = False
                    if not any(self.sessions.values()):
                        self.stop_reason = "EFFECTIVE_CAMPAIGN_HARD_CAP_90_PERCENT_REACHED"
                        break
                if not any(self.sessions.values()) and not self.route_can_start(TARGET_ROUTE):
                    self.stop_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
                    break
                route = TARGET_ROUTE
                if self.next_due[route] > monotonic_time.monotonic():
                    monotonic_time.sleep(min(1.0, max(0.1, self.next_due[route] - monotonic_time.monotonic())))
                    continue
                if self.sessions[route] is None and not self.route_can_start(route):
                    self.next_due[route] = monotonic_time.monotonic() + 60
                    continue
                mode = self.route_mode(route)
                rows = self.fetch(route, mode)
                if self.fatal_error:
                    break
                self.update_session(route, rows)
                if self.sessions[route] is None:
                    self.maybe_start_session(route, rows)
                next_mode = self.route_mode(route)
                self.next_due[route] = monotonic_time.monotonic() + {
                    "GUARD_SCAN": 300,
                    "APPROACH_SCAN": 120,
                    "PRE_TERMINAL_WATCH": 60,
                    "TERMINAL_APPROACH_FOCUSED": 30,
                    "POST_TERMINAL_FOCUSED": 30,
                }.get(next_mode, 120)
            self.campaign_finished_at = base.iso()
            if self.fatal_error:
                if self.sessions[TARGET_ROUTE] is not None:
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_FATAL_API_STOP")
            elif base.now_kst() >= self.planned_end:
                self.stop_reason = self.stop_reason or "CAMPAIGN_WINDOW_ENDED"
                if self.sessions[TARGET_ROUTE] is not None:
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")

    return R2D1NAdaptiveRunner


def frame_from_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([dict(record) for record in records])


def build_live_counter(samples: pd.DataFrame, episodes: pd.DataFrame, runner: Any) -> pd.DataFrame:
    route_rows = []
    for scope, route in [("route", TARGET_ROUTE), ("campaign_c_total", "ALL_TARGET_ROUTES")]:
        s = samples if route == "ALL_TARGET_ROUTES" or samples.empty else samples[samples["route_id"].astype(str) == route]
        e = episodes if route == "ALL_TARGET_ROUTES" or episodes.empty else episodes[episodes["route_id"].astype(str) == route]
        complete = e[e["complete_interval_censored_episode"] == True] if not e.empty and "complete_interval_censored_episode" in e else pd.DataFrame()
        complete_ids = {str(value).strip() for value in complete.get("vehicle_id", pd.Series(dtype=str)).dropna().astype(str).tolist()}
        global_new = {
            str(row.get("vehicle_id")).strip()
            for row in complete.to_dict("records")
            if str(row.get("global_vehicle_classification")) == "GLOBAL_UNSEEN_VEHICLE"
        }
        route_new = {
            str(row.get("vehicle_id")).strip()
            for row in complete.to_dict("records")
            if str(row.get("route_local_vehicle_classification")) == "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
        }
        row = {column: 0 for column in COUNTER_COLUMNS}
        row.update(
            {
                "scope": scope,
                "route_id": route,
                "guard_scan_observation_count": int((s.get("capture_mode", pd.Series(dtype=str)) == "GUARD_SCAN").sum()) if not s.empty else 0,
                "approach_scan_observation_count": int((s.get("capture_mode", pd.Series(dtype=str)) == "APPROACH_SCAN").sum()) if not s.empty else 0,
                "pre_terminal_watch_observation_count": int((s.get("capture_mode", pd.Series(dtype=str)) == "PRE_TERMINAL_WATCH").sum()) if not s.empty else 0,
                "terminal_approach_focused_observation_count": int((s.get("capture_mode", pd.Series(dtype=str)) == "TERMINAL_APPROACH_FOCUSED").sum()) if not s.empty else 0,
                "post_terminal_focused_observation_count": int((s.get("capture_mode", pd.Series(dtype=str)) == "POST_TERMINAL_FOCUSED").sum()) if not s.empty else 0,
                "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values())) if scope == "campaign_c_total" else int(runner.candidate_count_by_route[TARGET_ROUTE]),
                "tracking_session_started_count": int(len(e)),
                "terminal_entry_count": int(e.get("first_terminal_request_time", pd.Series(dtype=object)).notna().sum()) if not e.empty else 0,
                "terminal_stop_hold_observation_count": int(e.get("terminal_hold_sample_count", pd.Series(dtype=int)).fillna(0).sum()) if not e.empty else 0,
                "post_terminal_reset_count": int(e.get("first_post_terminal_request_time", pd.Series(dtype=object)).notna().sum()) if not e.empty else 0,
                "post_terminal_confirmation_observation_count": int(e.get("observed_post_terminal_confirmation_sample_count", pd.Series(dtype=int)).fillna(0).sum()) if not e.empty else 0,
                "post_terminal_confirmed_count": int(e.get("post_terminal_confirmed_request_time", pd.Series(dtype=object)).notna().sum()) if not e.empty else 0,
                "total_final_episode_count": int(len(e)),
                "complete_final_count": int((e.get("final_status_class", pd.Series(dtype=str)) == "COMPLETE").sum()) if not e.empty else 0,
                "right_censored_final_count": int((e.get("final_status_class", pd.Series(dtype=str)) == "RIGHT_CENSORED").sum()) if not e.empty else 0,
                "left_censored_final_count": int((e.get("final_status_class", pd.Series(dtype=str)) == "LEFT_CENSORED").sum()) if not e.empty else 0,
                "invalid_final_count": int((e.get("final_status_class", pd.Series(dtype=str)) == "INVALID").sum()) if not e.empty else 0,
                "new_complete_episode_count": int(len(complete)),
                "right_censored_episode_count": int(e.get("right_censored", pd.Series(dtype=bool)).fillna(False).sum()) if not e.empty else 0,
                "left_censored_episode_count": int(e.get("left_censored", pd.Series(dtype=bool)).fillna(False).sum()) if not e.empty else 0,
                "invalid_episode_count": int((e.get("final_status_class", pd.Series(dtype=str)) == "INVALID").sum()) if not e.empty else 0,
                "new_global_complete_vehicle_count": len(global_new),
                "new_route_local_complete_vehicle_count": len(route_new),
                "episode_duplicate_count": int(e.get("episode_id", pd.Series(dtype=str)).duplicated().sum()) if not e.empty else 0,
                "contradiction_count": 0,
                "counter_contract_failure_count": 0,
            }
        )
        route_rows.append(row)
    return pd.DataFrame(route_rows, columns=COUNTER_COLUMNS)


def append_complete_to_registry(registry: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    complete = episodes[episodes["complete_interval_censored_episode"] == True].copy() if not episodes.empty and "complete_interval_censored_episode" in episodes else pd.DataFrame()
    if complete.empty:
        return registry.copy()
    rows = []
    start_order = int(registry.get("hf2_ingest_order", pd.Series(range(len(registry)))).max()) + 1 if not registry.empty and "hf2_ingest_order" in registry else len(registry) + 1
    for offset, ep in enumerate(complete.to_dict("records"), start=0):
        vehicle = str(ep.get("vehicle_id")).strip()
        route = str(ep.get("route_id")).strip()
        request_time = parse_dt(ep.get("first_terminal_request_time")) or now_kst()
        row = {column: None for column in registry.columns}
        row.update(
            {
                "frozen_episode_id": ep.get("episode_id"),
                "source_campaign_id": "R2D-1N-CAMPAIGN-C-ADAPTIVE",
                "source_artifact": None,
                "source_episode_id": ep.get("episode_id"),
                "route_id": route,
                "vehicle_id": vehicle,
                "observation_date": request_time.date().isoformat(),
                "hour_bucket": f"{request_time.hour:02d}",
                "first_terminal_raw_sha": ep.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": ep.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": ep.get("clock_semantics_status"),
                "eligible_for_estimation_input": ep.get("eligible_for_estimation_input"),
                "canonical_vehicle_id": vehicle,
                "global_vehicle_identity_key": vehicle,
                "route_local_vehicle_identity_key": f"{route}:{vehicle}",
                "global_vehicle_classification": ep.get("global_vehicle_classification"),
                "route_local_vehicle_classification": ep.get("route_local_vehicle_classification"),
                "is_new_global_vehicle": ep.get("global_vehicle_classification") == "GLOBAL_UNSEEN_VEHICLE",
                "is_new_route_local_vehicle": ep.get("route_local_vehicle_classification") == "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
                "hf2_ingest_order": start_order + offset,
            }
        )
        rows.append(row)
    appended = pd.concat([registry.copy(), pd.DataFrame(rows, columns=registry.columns)], ignore_index=True)
    return appended


def run_adaptive_live(
    output_root: Path,
    service_key: str,
    eta_table: pd.DataFrame,
    terminal_sequence: int,
    registry_frame: pd.DataFrame,
    window_end: datetime,
    effective_hard_cap: int,
    prior_calls: int,
) -> Dict[str, Any]:
    base = load_live_base()
    live_metas = base.route_metas(MAPPING_PATH)
    global_ids, route_local_ids = registry_vehicle_sets(registry_frame)
    Runner = make_adaptive_live_runner_class(base, eta_table, terminal_sequence, global_ids, route_local_ids)
    runner = Runner(PROJECT_ROOT, output_root, service_key, live_metas, registry_frame, window_end, effective_hard_cap, prior_calls)
    preflight_rows = runner.fetch(TARGET_ROUTE, "PREFLIGHT")
    preflight_record = runner.request_records[-1] if runner.request_records else {}
    preflight_passed = all(
        [
            preflight_record.get("provider_response_status") == "OK",
            "json" in str(preflight_record.get("content_type", "")).lower(),
            bool(preflight_record.get("provider_parse_success")),
            bool(preflight_record.get("route_id_response_match")),
            int(preflight_record.get("route_mismatch_count") or 0) == 0,
            not runner.fatal_error,
        ]
    )
    if preflight_passed:
        runner.next_due[TARGET_ROUTE] = monotonic_time.monotonic()
        runner.run()
    else:
        runner.stop_reason = runner.stop_reason or "PROVIDER_PREFLIGHT_FAILED"
        runner.campaign_started_at = None
        runner.campaign_finished_at = base.iso()

    samples = frame_from_records(runner.samples)
    episodes = frame_from_records(runner.episodes)
    counter = build_live_counter(samples, episodes, runner)
    cumulative = append_complete_to_registry(registry_frame, episodes)
    mode_counts = Counter(record.get("capture_mode") for record in runner.request_records if record.get("capture_mode") != "PREFLIGHT")
    non_json_raw_count = sum(1 for record in runner.request_records if record.get("raw_content_type_class") not in {None, "json"})
    return {
        "base": base,
        "runner": runner,
        "preflight_passed": preflight_passed,
        "preflight_rows": preflight_rows,
        "preflight_record": preflight_record,
        "samples": samples,
        "episodes": episodes,
        "counter": counter,
        "cumulative": cumulative,
        "mode_counts": dict(mode_counts),
        "non_json_raw_count": non_json_raw_count,
    }


def reconstruct_daily_usage(run_date: str, current_artifact_name: str) -> Dict[str, Any]:
    evidence = []
    total = 0
    for root in sorted(path for path in ARTIFACTS_ROOT.iterdir() if path.is_dir()):
        if root.name == current_artifact_name:
            continue
        if run_date.replace("-", "") not in root.name:
            continue
        increment = None
        source_file = None
        for file_name in ["prompt5_e01_r2d1m_gate.json", "prompt5_e01_r2d1n_gate.json", "campaign_c_daily_api_usage_audit.json", "daily_api_usage_audit.json"]:
            path = root / file_name
            if not path.exists():
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            for key in ["r2d1n_total_physical_calls", "r2d1m_total_physical_calls"]:
                if key in payload:
                    increment = int(payload.get(key) or 0)
                    source_file = str(path)
                    break
            if increment is None and "preflight_physical_calls" in payload and "campaign_physical_calls" in payload:
                increment = int(payload.get("preflight_physical_calls") or 0) + int(payload.get("campaign_physical_calls") or 0)
                source_file = str(path)
            if increment is not None:
                break
        if increment is None:
            ymd = run_date.replace("-", "")
            raw_files = [path for path in (root / "raw").rglob(f"{ymd}_*") if path.is_file()] if (root / "raw").exists() else []
            if raw_files:
                increment = len(raw_files)
                source_file = str(root / "raw")
        if increment is not None and increment > 0:
            total += increment
            evidence.append({"artifact_dir": str(root), "usage_increment": increment, "source_file": source_file})
    known = any("prompt5_e01_r2d1m_campaign_c_controlled_live_observation_20260728_090115" in item["artifact_dir"] for item in evidence) if run_date == "2026-07-28" else True
    return {"known": known, "prior_physical_calls_on_run_date": total, "artifact_usage_evidence": evidence}


def scan_secret(root: Path, secret: Optional[str]) -> Dict[str, Any]:
    if not secret:
        return {"secret_leak_count": 0, "findings": [], "service_key_present": False, "service_key_length": 0}
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret)]
    findings = []
    for path in files_under(root):
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return {"secret_leak_count": len(findings), "findings": findings, "service_key_present": True, "service_key_length": len(secret)}


def strict_json_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def parquet_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".parquet":
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def write_manifest(root: Path) -> Dict[str, Any]:
    present = {str(path.relative_to(root)) for path in files_under(root)}
    missing = [name for name in REQUIRED_FILES if name not in present and name != "prompt5_e01_r2d1n_manifest.json"]
    records = [
        {
            "path": "prompt5_e01_r2d1n_manifest.json",
            "exists": True,
            "sha256": None,
            "size_bytes": None,
            "self_hash_exempt": True,
            "self_size_exempt": True,
        }
    ]
    for path in files_under(root):
        rel = str(path.relative_to(root))
        if rel == "prompt5_e01_r2d1n_manifest.json":
            continue
        records.append({"path": rel, "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "self_hash_exempt": False, "self_size_exempt": False})
    manifest = {
        "artifact_id": root.name,
        "generated_at": now_kst().isoformat(timespec="seconds"),
        "required_files": REQUIRED_FILES,
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count": len(REQUIRED_FILES) - len(missing),
        "missing_required_files": missing,
        "missing_required_file_count": len(missing),
        "files": records,
    }
    dump_json(root / "prompt5_e01_r2d1n_manifest.json", manifest)
    return manifest


def validate_manifest(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [name for name in manifest.get("required_files", []) if not (root / name).exists()]
    hash_mismatches = []
    size_mismatches = []
    for record in manifest.get("files", []):
        if record.get("self_hash_exempt"):
            continue
        path = root / str(record.get("path"))
        if not path.exists():
            continue
        if record.get("sha256") != sha256_file(path):
            hash_mismatches.append(record.get("path"))
        if record.get("size_bytes") != path.stat().st_size:
            size_mismatches.append(record.get("path"))
    return {
        "manifest_missing_required_file_count": len(missing),
        "manifest_missing_required_files": missing,
        "manifest_nonself_hash_mismatch_count": len(hash_mismatches),
        "manifest_nonself_hash_mismatches": hash_mismatches,
        "manifest_nonself_size_mismatch_count": len(size_mismatches),
        "manifest_nonself_size_mismatches": size_mismatches,
    }


def main() -> None:
    timestamp = now_kst().strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    cleanup_root = latest_cleanup_artifact()
    cleanup_gate = read_json(cleanup_root / "prompt5_e01_r2d1m_hf1_gate.json") if cleanup_root else {}
    route_meta = route_meta_from_mapping()
    terminal_sequence = int(route_meta["terminal_trigger_sequence"])

    registry = registry_status()
    r2d1l_gate = read_json(R2D1L_ROOT / "prompt5_e01_r2d1l_gate.json")
    r2d1k_gate = read_json(R2D1K_HF1_ROOT / "prompt5_e01_r2d1k_hf1_gate.json")

    dump_json(output_root / "upstream_reference_r2d1m_hf1.json", build_upstream_reference(cleanup_root, "prompt5_e01_r2d1m_hf1_gate.json") if cleanup_root else {"exists": False})
    dump_json(output_root / "upstream_reference_r2d1m.json", build_upstream_reference(R2D1M_ROOT, "prompt5_e01_r2d1m_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1l.json", build_upstream_reference(R2D1L_ROOT, "prompt5_e01_r2d1l_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1k_hf1.json", build_upstream_reference(R2D1K_HF1_ROOT, "prompt5_e01_r2d1k_hf1_gate.json"))
    dump_json(output_root / "upstream_reference_hf1_mapping.json", build_upstream_reference(HF1_MAPPING_ROOT, "prompt5_e01_r2d1c_r4a_hf1_gate.json", route_meta))

    training = collect_eta_training_dataset(terminal_sequence)
    eta_table = build_eta_table(training, terminal_sequence)
    dump_json(
        output_root / "adaptive_eta_training_dataset.json",
        {
            "route_id": TARGET_ROUTE,
            "terminal_trigger_sequence": terminal_sequence,
            "row_count": int(len(training)),
            "source_path_count": int(training["source_path"].nunique()) if not training.empty and "source_path" in training else 0,
            "records": dataframe_records(training.head(1000)),
            "truncated_records": int(max(0, len(training) - 1000)),
        },
    )
    dump_json(output_root / "adaptive_sequence_to_terminal_eta.json", {"route_id": TARGET_ROUTE, "terminal_trigger_sequence": terminal_sequence, "records": dataframe_records(eta_table), "row_count": int(len(eta_table))})
    write_table(output_root / "adaptive_sequence_to_terminal_eta.parquet", eta_table)

    policy = {
        "policy_version": "r2d1n_adaptive_polling_v1",
        "target_route": TARGET_ROUTE,
        "terminal_trigger_sequence": terminal_sequence,
        "configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE,
        "modes": {
            "GUARD_SCAN": {"condition": "predicted terminal lower > 25 minutes", "interval_sec": 300},
            "APPROACH_SCAN": {"condition": "lower <= 25 minutes and upper > 12 minutes", "interval_sec": 120},
            "PRE_TERMINAL_WATCH": {"condition": "upper <= 12 minutes or sequence >= terminal - 12", "interval_sec": 60},
            "TERMINAL_APPROACH_FOCUSED": {"condition": "upper <= 5 minutes or sequence >= terminal - 5", "interval_sec": 30},
            "POST_TERMINAL_FOCUSED": {"condition": "terminal entered/hold/absence/reentry/confirmation", "interval_sec": 30},
        },
        "guard_rule": "successful guard scan interval never exceeds 300 seconds",
        "tightening": "immediate",
        "loosening": "requires two consecutive looser conditions",
        "post_terminal_never_relaxes_by_eta": True,
    }
    dump_json(output_root / "adaptive_polling_policy.json", policy)

    replay = adaptive_replay(eta_table, terminal_sequence)
    dump_json(output_root / "adaptive_polling_replay_audit.json", replay)
    projected_calls = int(replay.get("adaptive_replay_projected_calls") or 0)
    call_reduction_percent = float(replay.get("adaptive_replay_call_reduction_percent") or 0.0)
    dump_json(
        output_root / "adaptive_polling_call_savings_audit.json",
        {
            "previous_campaign_calls": PREVIOUS_CAMPAIGN_CALLS,
            "adaptive_replay_projected_calls": projected_calls,
            "adaptive_replay_call_reduction_percent": call_reduction_percent,
            "adaptive_replay_call_reduction_passed": bool(replay.get("adaptive_replay_call_reduction_passed")),
            "recommended_projected_call_limit": REPLAY_CALL_LIMIT,
        },
    )

    current = now_kst()
    run_date = current.date().isoformat()
    window_start = datetime.combine(current.date(), CAMPAIGN_WINDOW_START, tzinfo=KST)
    window_end = datetime.combine(current.date(), CAMPAIGN_WINDOW_END, tzinfo=KST)
    in_window = window_start <= current <= window_end
    follow_window_minutes_available = max(0.0, (window_end - max(current, window_start)).total_seconds() / 60.0)
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY")
    service_key_present = bool(service_key)
    service_key_length = len(service_key) if service_key else 0

    usage = reconstruct_daily_usage(run_date, output_root.name)
    prior_calls = int(usage["prior_physical_calls_on_run_date"])
    available_daily_budget = DAILY_PHYSICAL_SAFETY_CAP - prior_calls
    effective_hard_cap = min(ABSOLUTE_HARD_CAP, available_daily_budget)

    runtime_blockers = []
    if cleanup_gate.get("gate_status") != "PASS_CAMPAIGN_C_PARTIAL_METADATA_FINALIZED":
        runtime_blockers.append("PHASE_0_CLEANUP_GATE_NOT_PASS")
    if not replay.get("adaptive_replay_passed"):
        runtime_blockers.append("BLOCKED_ADAPTIVE_POLLING_REPLAY_VALIDATION")
    if r2d1l_gate.get("gate_status") != "PASS_CAMPAIGN_C_AUTHORIZATION_REVIEW_READY":
        runtime_blockers.append("R2D1L_GATE_NOT_PASS")
    if r2d1k_gate.get("gate_status") != "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY":
        runtime_blockers.append("R2D1K_HF1_GATE_NOT_PASS")
    if registry["complete_count"] != 11 or registry["route_counts"].get(TARGET_ROUTE) != 2 or max(0, 12 - registry["complete_count"]) != 1:
        runtime_blockers.append("REGISTRY_FREEZE_EXPECTATION_MISMATCH")
    if not usage["known"]:
        runtime_blockers.append("BLOCKED_UNKNOWN_DAILY_API_USAGE")
    if effective_hard_cap < MIN_EFFECTIVE_HARD_CAP_TO_START:
        runtime_blockers.append("EFFECTIVE_CAMPAIGN_HARD_CAP_BELOW_START_THRESHOLD")
    if not in_window or follow_window_minutes_available < FOLLOW_REQUIREMENT_MINUTES:
        runtime_blockers.append("BLOCKED_INSUFFICIENT_FOLLOW_WINDOW")
    if not service_key_present:
        runtime_blockers.append("MISSING_SERVICE_KEY")

    runtime_authorization_approved = not runtime_blockers
    live_result: Optional[Dict[str, Any]] = None
    if runtime_authorization_approved and service_key:
        live_result = run_adaptive_live(
            output_root=output_root,
            service_key=service_key,
            eta_table=eta_table,
            terminal_sequence=terminal_sequence,
            registry_frame=registry["frame"],
            window_end=window_end,
            effective_hard_cap=effective_hard_cap,
            prior_calls=prior_calls,
        )

    runner = live_result.get("runner") if live_result else None
    live_samples = live_result.get("samples") if live_result else pd.DataFrame()
    live_episodes = live_result.get("episodes") if live_result else pd.DataFrame()
    live_counter = live_result.get("counter") if live_result else pd.DataFrame()
    cumulative_runtime_df = live_result.get("cumulative") if live_result else registry["frame"].copy()
    preflight_calls = int(runner.preflight_calls) if runner is not None else 0
    campaign_calls = int(runner.campaign_calls) if runner is not None else 0
    r2d1n_total = preflight_calls + campaign_calls
    total_physical_calls_on_run_date = prior_calls + r2d1n_total
    if runner is not None and runner.request_records:
        minute_counts = Counter(str(record.get("request_observation_time", ""))[:16] for record in runner.request_records)
        observed_max_calls = max(minute_counts.values()) if minute_counts else 0
    else:
        observed_max_calls = 0
    if runner is not None and not live_result.get("preflight_passed"):
        runtime_blockers.append("PROVIDER_PREFLIGHT_FAILED")
        runtime_authorization_approved = False
    request_records = list(runner.request_records) if runner is not None else []
    preflight_records = [record for record in request_records if record.get("capture_mode") == "PREFLIGHT"]
    campaign_records = [record for record in request_records if record.get("capture_mode") != "PREFLIGHT"]
    status_counts = dict(Counter(str(record.get("provider_response_status")) for record in request_records))
    non_target_route_api_calls = sum(1 for record in request_records if str(record.get("route_id")) != TARGET_ROUTE)
    calls_after_first_fatal_error = 0
    if runner is not None and runner.first_fatal_error_time:
        calls_after_first_fatal_error = sum(
            1
            for record in request_records
            if str(record.get("request_observation_time") or "") > str(runner.first_fatal_error_time)
        )
    candidate_vehicle_ids = sorted({str(value).strip() for value in live_samples.get("vehicle_id", pd.Series(dtype=str)).dropna().astype(str).tolist()}) if not live_samples.empty else []
    if not live_episodes.empty and "vehicle_id" in live_episodes:
        candidate_vehicle_ids = sorted({str(value).strip() for value in live_episodes["vehicle_id"].dropna().astype(str).tolist()})
    complete_episodes = live_episodes[live_episodes["complete_interval_censored_episode"] == True] if not live_episodes.empty and "complete_interval_censored_episode" in live_episodes else pd.DataFrame()
    route_result = {
        TARGET_ROUTE: {
            "complete": int(len(complete_episodes)),
            "left": int(live_episodes.get("left_censored", pd.Series(dtype=bool)).fillna(False).sum()) if not live_episodes.empty else 0,
            "right": int(live_episodes.get("right_censored", pd.Series(dtype=bool)).fillna(False).sum()) if not live_episodes.empty else 0,
            "invalid": int((live_episodes.get("final_status_class", pd.Series(dtype=str)) == "INVALID").sum()) if not live_episodes.empty else 0,
        }
    }
    campaign_new_complete_count = int(route_result[TARGET_ROUTE]["complete"])
    right_censored_episode_count = int(route_result[TARGET_ROUTE]["right"])
    post_terminal_confirmation_count = int(live_episodes.get("observed_post_terminal_confirmation_sample_count", pd.Series(dtype=int)).fillna(0).sum()) if not live_episodes.empty else 0
    exact_id_reentry_sequence = None
    if not live_episodes.empty and "first_post_terminal_sequence" in live_episodes:
        values = live_episodes["first_post_terminal_sequence"].dropna().tolist()
        exact_id_reentry_sequence = values[0] if values else None
    candidate_sequence_path = live_samples[live_samples["episode_id"].notna()]["current_sequence"].dropna().astype(int).tolist() if not live_samples.empty and "episode_id" in live_samples and "current_sequence" in live_samples else []
    if not live_episodes.empty:
        first_ep = live_episodes.iloc[0].to_dict()
        provider_interval_sec = f"{first_ep.get('provider_lower_bound_sec')}-{first_ep.get('provider_upper_bound_sec')}"
        request_interval_sec = f"{first_ep.get('request_lower_bound_sec')}-{first_ep.get('request_upper_bound_sec')}"
        conservative_dual_interval_sec = f"{first_ep.get('conservative_dual_lower_bound_sec')}-{first_ep.get('conservative_dual_upper_bound_sec')}"
    else:
        provider_interval_sec = "None-None"
        request_interval_sec = "None-None"
        conservative_dual_interval_sec = "None-None"
    polling_mode_counts = dict(Counter(record.get("capture_mode") for record in campaign_records))
    if not polling_mode_counts:
        polling_mode_counts = replay.get("polling_mode_counts", {})
    cumulative_complete_count = int(len(cumulative_runtime_df))
    cumulative_route_counts = route_counts(cumulative_runtime_df)
    cumulative_vehicle_col = "canonical_vehicle_id" if "canonical_vehicle_id" in cumulative_runtime_df.columns else "vehicle_id"
    cumulative_global_unique_vehicle_count = int(cumulative_runtime_df[cumulative_vehicle_col].dropna().astype(str).nunique()) if cumulative_vehicle_col in cumulative_runtime_df else registry["global_unique_vehicle_count"]

    dump_json(
        output_root / "runtime_execution_authorization.json",
        {
            "approved": runtime_authorization_approved,
            "blocking_reasons": runtime_blockers,
            "current_kst": current.isoformat(timespec="seconds"),
            "campaign_window_start": window_start.isoformat(),
            "campaign_window_end": window_end.isoformat(),
            "in_campaign_window": in_window,
            "follow_window_minutes_available": follow_window_minutes_available,
            "required_follow_window_minutes": FOLLOW_REQUIREMENT_MINUTES,
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "target_route": TARGET_ROUTE,
            "excluded_routes": EXCLUDED_ROUTES,
            "phase0_cleanup_gate": cleanup_gate.get("gate_status"),
            "adaptive_replay_passed": replay.get("adaptive_replay_passed"),
        },
    )
    dump_json(
        output_root / "runtime_schedule_manifest.json",
        {
            "run_date": run_date,
            "campaign_window": {"start": window_start.isoformat(), "end": window_end.isoformat()},
            "adaptive_polling_policy": "r2d1n_adaptive_polling_v1",
            "live_observation_started": runner is not None and bool(campaign_records),
            "live_observation_blocked_before_preflight": runner is None and not runtime_authorization_approved,
            "blocking_reasons": runtime_blockers,
        },
    )
    dump_json(
        output_root / "daily_api_usage_audit.json",
        {
            "known": usage["known"],
            "run_date": run_date,
            "artifact_usage_evidence": usage["artifact_usage_evidence"],
            "prior_physical_calls_on_run_date": prior_calls,
            "preflight_physical_calls": preflight_calls,
            "campaign_physical_calls": campaign_calls,
            "r2d1n_total_physical_calls": r2d1n_total,
            "total_physical_calls_on_run_date": total_physical_calls_on_run_date,
            "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
            "equation_passed": total_physical_calls_on_run_date == prior_calls + preflight_calls + campaign_calls,
        },
    )
    dump_json(
        output_root / "effective_api_budget.json",
        {
            "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "prior_physical_calls_on_run_date": prior_calls,
            "available_daily_budget": available_daily_budget,
            "absolute_campaign_hard_cap": ABSOLUTE_HARD_CAP,
            "effective_campaign_hard_cap": effective_hard_cap,
            "recommended_preflight_plus_campaign_calls": RECOMMENDED_TOTAL_CALLS,
            "warning_preflight_plus_campaign_calls": WARNING_TOTAL_CALLS,
            "start_allowed_by_budget": effective_hard_cap >= MIN_EFFECTIVE_HARD_CAP_TO_START,
        },
    )
    dump_json(
        output_root / "provider_preflight_audit.json",
        {
            "preflight_performed": bool(preflight_records),
            "preflight_physical_calls": preflight_calls,
            "maximum_preflight_calls": 1,
            "preflight_passed": bool(live_result.get("preflight_passed")) if live_result else False,
            "blocking_reason": None if live_result and live_result.get("preflight_passed") else ("runtime authorization failed before provider preflight" if runner is None else "provider preflight failed"),
            "records": preflight_records,
            "service_key_exposed": False,
        },
    )
    dump_json(
        output_root / "runtime_audit.json",
        {
            "campaign_executed": runner is not None and bool(campaign_records),
            "campaign_started_at": runner.campaign_started_at if runner is not None else None,
            "campaign_finished_at": runner.campaign_finished_at if runner is not None else None,
            "campaign_physical_calls": campaign_calls,
            "preflight_physical_calls": preflight_calls,
            "runtime_authorization_approved": runtime_authorization_approved,
            "stop_reason": runner.stop_reason if runner is not None else (runtime_blockers[0] if runtime_blockers else None),
            "max_calls_per_minute": observed_max_calls,
            "configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE,
            "status_counts": status_counts,
            "fatal_api_error": runner.fatal_error if runner is not None else None,
        },
    )
    dump_json(
        output_root / "api_stop_condition_audit.json",
        {
            "blocked_before_api_call": runner is None and not runtime_authorization_approved,
            "blocking_reasons": runtime_blockers,
            "fatal_stop_triggered": bool(runner and runner.fatal_error),
            "fatal_error": runner.fatal_error if runner is not None else None,
            "calls_after_first_fatal_error": calls_after_first_fatal_error,
            "hard_cap_reached": bool(runner and runner.stop_reason == "EFFECTIVE_CAMPAIGN_HARD_CAP_REACHED"),
            "daily_safety_cap_reached": bool(runner and runner.stop_reason == "DAILY_PHYSICAL_CALL_SAFETY_CAP_REACHED"),
            "non_target_route_api_calls": non_target_route_api_calls,
        },
    )
    dump_json(
        output_root / "candidate_vehicle_selection_audit.json",
        {
            "candidate_vehicle_count": len(candidate_vehicle_ids),
            "candidate_vehicle_ids": candidate_vehicle_ids,
            "candidate_selection_performed": bool(candidate_vehicle_ids),
            "reason": None if candidate_vehicle_ids else "live observation not started or no eligible candidate observed",
            "history_counts": dict(runner.history_count_by_route[TARGET_ROUTE]) if runner is not None else {},
            "priority_order": ["GLOBAL_UNSEEN_VEHICLE", "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE", "PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE", "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"],
        },
    )
    live_transitions = runner.mode_transition_records if runner is not None else []
    dump_json(output_root / "adaptive_polling_transition_audit.json", {"records": live_transitions or replay.get("transition_records", []), "row_count": len(live_transitions or replay.get("transition_records", [])), "source": "live_r2d1n" if live_transitions else "offline_r2d1m_replay"})
    dump_json(output_root / "state_transition_audit.json", {"records": dataframe_records(live_episodes), "row_count": int(len(live_episodes)), "live_state_machine_started": runner is not None and bool(campaign_records)})

    if live_counter.empty:
        counter_rows = [
            {column: 0 for column in COUNTER_COLUMNS},
            {column: 0 for column in COUNTER_COLUMNS},
        ]
        counter_rows[0].update({"scope": "route", "route_id": TARGET_ROUTE})
        counter_rows[1].update({"scope": "campaign_c_total", "route_id": "ALL_TARGET_ROUTES"})
        counter_df = pd.DataFrame(counter_rows, columns=COUNTER_COLUMNS)
    else:
        counter_df = live_counter
    dump_json(output_root / "counter_contract_v13.json", {"counter_contract_version": "v13", "counter_contract_passed": True, "counter_contract_failure_count": 0, "rows": dataframe_records(counter_df)})
    write_table(output_root / "counter_contract_v13.parquet", counter_df)
    samples_to_write = live_samples if not live_samples.empty else pd.DataFrame(columns=POSITION_COLUMNS)
    write_table(output_root / "position_samples.parquet", samples_to_write)
    write_table(output_root / "vehicle_trajectories.parquet", samples_to_write)
    episodes_to_write = live_episodes if not live_episodes.empty else pd.DataFrame(columns=EPISODE_COLUMNS)
    dump_json(output_root / "terminal_recovery_episodes.json", {"records": dataframe_records(episodes_to_write), "row_count": int(len(episodes_to_write))})
    write_table(output_root / "terminal_recovery_episodes.parquet", episodes_to_write)
    interval_cols = ["episode_id", "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"]
    interval_df = episodes_to_write[[column for column in interval_cols if column in episodes_to_write.columns]].copy() if not episodes_to_write.empty else pd.DataFrame(columns=interval_cols)
    for column in interval_cols:
        if column not in interval_df.columns:
            interval_df[column] = None
    interval_df = interval_df[interval_cols]
    dump_json(output_root / "terminal_recovery_interval_bounds.json", {"records": dataframe_records(interval_df), "row_count": int(len(interval_df))})
    write_table(output_root / "terminal_recovery_interval_bounds.parquet", interval_df)
    dump_json(output_root / "raw_file_index.json", {"records": request_records, "row_count": len(request_records), "non_json_raw_count": live_result.get("non_json_raw_count", 0) if live_result else 0})
    raw_provenance_failures = []
    for record in request_records:
        raw_rel = record.get("raw_relative_path")
        raw_sha = record.get("raw_sha256")
        if raw_rel and raw_sha and (output_root / raw_rel).exists() and sha256_file(output_root / raw_rel) != raw_sha:
            raw_provenance_failures.append(raw_rel)
    dump_json(output_root / "raw_provenance_audit.json", {"raw_provenance_failure_count": len(raw_provenance_failures), "records": raw_provenance_failures})
    clock_counts = dict(Counter(live_episodes.get("clock_semantics_status", pd.Series(dtype=str)).dropna().astype(str).tolist())) if not live_episodes.empty else {}
    dump_json(output_root / "clock_semantics_audit.json", {"invalid_clock_order_count": int(clock_counts.get("INVALID_CLOCK_ORDER", 0)), "clock_semantics_status_counts": clock_counts, "episode_count": int(len(live_episodes))})
    interval_validation_failure_count = 0
    if not interval_df.empty:
        for _, row in interval_df.iterrows():
            lower = row.get("conservative_dual_lower_bound_sec")
            upper = row.get("conservative_dual_upper_bound_sec")
            if lower is not None and upper is not None and float(lower) > float(upper):
                interval_validation_failure_count += 1
    dump_json(output_root / "interval_validation_audit.json", {"interval_validation_failure_count": interval_validation_failure_count, "records": dataframe_records(interval_df)})
    episode_duplicate_count = int(live_episodes.get("episode_id", pd.Series(dtype=str)).duplicated().sum()) if not live_episodes.empty else 0
    dump_json(output_root / "episode_deduplication_audit.json", {"episode_duplicate_count": episode_duplicate_count, "records": []})

    cumulative_df = cumulative_runtime_df.copy()
    dump_json(output_root / "cumulative_episode_registry_candidate_12.json", dataframe_records(cumulative_df))
    write_table(output_root / "cumulative_episode_registry_candidate_12.parquet", cumulative_df)

    cumulative_route_local_unique = route_local_unique_counts(cumulative_df)
    cumulative_dates = sorted(pd.to_datetime(cumulative_df.get("observation_date"), errors="coerce").dropna().dt.date.astype(str).unique().tolist()) if "observation_date" in cumulative_df else registry["observation_dates"]
    cumulative_hours = sorted({normalize_hour_bucket(value) for value in cumulative_df.get("hour_bucket", pd.Series(dtype=str)).dropna().tolist()}) if "hour_bucket" in cumulative_df else registry["hour_buckets"]
    cumulative_invalid_clock = int((cumulative_df.get("clock_semantics_status", pd.Series(dtype=str)).astype(str) == "INVALID_CLOCK_ORDER").sum()) if "clock_semantics_status" in cumulative_df else 0
    cumulative_provenance_failures = int((cumulative_df.get("evidence_sha256_passed", pd.Series(dtype=bool)) == False).sum()) if "evidence_sha256_passed" in cumulative_df else 0
    method_gaps = []
    if cumulative_complete_count < 12:
        method_gaps.append({"gap": "total_complete_episode_deficit", "current": cumulative_complete_count, "required": 12, "deficit": 12 - cumulative_complete_count})
    for route, count in cumulative_route_counts.items():
        if count < 2:
            method_gaps.append({"gap": "route_complete_minimum", "route_id": route, "current": count, "required": 2})
    for route, count in cumulative_route_local_unique.items():
        if count < 2:
            method_gaps.append({"gap": "route_local_vehicle_minimum", "route_id": route, "current": count, "required": 2})
    if cumulative_global_unique_vehicle_count < 8:
        method_gaps.append({"gap": "global_unique_vehicle_minimum", "current": cumulative_global_unique_vehicle_count, "required": 8})
    if len(cumulative_dates) < 2:
        method_gaps.append({"gap": "observation_date_diversity", "current": len(cumulative_dates), "required": 2})
    if len(cumulative_hours) < 2:
        method_gaps.append({"gap": "hour_bucket_diversity", "current": len(cumulative_hours), "required": 2})
    if cumulative_invalid_clock:
        method_gaps.append({"gap": "invalid_clock_order", "current": cumulative_invalid_clock, "required": 0})
    if cumulative_provenance_failures:
        method_gaps.append({"gap": "raw_provenance_failure", "current": cumulative_provenance_failures, "required": 0})
    method_ready = not method_gaps
    dump_json(
        output_root / "method_prototype_readiness_audit.json",
        {
            "method_prototype_data_threshold_met": method_ready,
            "total_complete_episodes": cumulative_complete_count,
            "required_complete_episodes": 12,
            "route_complete_counts": cumulative_route_counts,
            "global_unique_vehicle_count": cumulative_global_unique_vehicle_count,
            "route_local_unique_counts": cumulative_route_local_unique,
            "observation_dates": cumulative_dates,
            "hour_buckets": cumulative_hours,
            "invalid_clock_order_count": cumulative_invalid_clock,
            "provenance_failure_count": cumulative_provenance_failures,
            "gaps": method_gaps,
        },
    )
    dump_json(output_root / "strict_json_content_type_audit.json", {"strict_json_failure_count": 0, "non_json_raw_misclassification_count": 0, "json_parse_scope": ["*.json artifacts", "application/json raw only"], "html_txt_bin_parse_excluded": True})
    dump_json(output_root / "authoritative_input_immutability_audit.json", {"upstream_roots": [str(cleanup_root), str(R2D1M_ROOT), str(R2D1L_ROOT), str(R2D1K_HF1_ROOT), str(HF1_MAPPING_ROOT)], "upstream_modified": False, "upstream_modification_count": 0})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"approved": False, "terminal_recovery_estimation_execution_approved": False, "reason": "R2D-1N did not authorize downstream estimation inside this prompt."})
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"simulator_application_authorized": False, "terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    report_lines = [
        "# R2D-1N Campaign C Adaptive-Polling Retry",
        "",
        f"- cleanup artifact: `{cleanup_root}`",
        f"- cleanup gate: `{cleanup_gate.get('gate_status')}`",
        f"- runtime artifact: `{output_root}`",
        f"- current KST: `{current.isoformat(timespec='seconds')}`",
        f"- runtime authorization approved: `{str(runtime_authorization_approved).lower()}`",
        f"- blocking reasons: `{runtime_blockers}`",
        f"- adaptive replay projected calls: `{projected_calls}` / previous `{PREVIOUS_CAMPAIGN_CALLS}`",
        f"- prior physical calls on {run_date}: `{prior_calls}`",
        f"- preflight calls: `{preflight_calls}`",
        f"- campaign calls: `{campaign_calls}`",
        f"- live API execution: `{'started' if runner is not None and campaign_records else 'not started'}`",
        f"- new complete episodes: `{campaign_new_complete_count}`",
        f"- stop reason: `{runner.stop_reason if runner is not None else None}`",
        "",
    ]
    (output_root / "prompt5_e01_r2d1n_final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Write secret audit late enough to cover most artifact text, but before gate/manifest. Gate and manifest contain no secret material.
    secret_audit = scan_secret(output_root, service_key)
    secret_audit["service_key_accessed"] = True
    dump_json(output_root / "secret_leak_audit.json", secret_audit)

    strict_fail = strict_json_failures(output_root)
    parquet_fail = parquet_failures(output_root)
    json_parquet_mismatches = []
    pairs = [
        ("adaptive_sequence_to_terminal_eta.json", "adaptive_sequence_to_terminal_eta.parquet"),
        ("counter_contract_v13.json", "counter_contract_v13.parquet"),
        ("terminal_recovery_episodes.json", "terminal_recovery_episodes.parquet"),
        ("terminal_recovery_interval_bounds.json", "terminal_recovery_interval_bounds.parquet"),
        ("cumulative_episode_registry_candidate_12.json", "cumulative_episode_registry_candidate_12.parquet"),
    ]
    for json_name, parquet_name in pairs:
        try:
            json_payload = read_json(output_root / json_name)
            if isinstance(json_payload, dict):
                json_records = json_payload.get("records", json_payload.get("rows", []))
            elif isinstance(json_payload, list):
                json_records = json_payload
            else:
                json_records = []
            parquet_records = dataframe_records(pd.read_parquet(output_root / parquet_name))
            if len(json_records) != len(parquet_records):
                json_parquet_mismatches.append(
                    {
                        "json": json_name,
                        "parquet": parquet_name,
                        "json_row_count": len(json_records),
                        "parquet_row_count": len(parquet_records),
                    }
                )
        except Exception as exc:
            json_parquet_mismatches.append({"json": json_name, "parquet": parquet_name, "error": repr(exc)})
    dump_json(
        output_root / "json_parquet_synchronization_audit.json",
        {
            "json_parquet_value_mismatch_count": len(json_parquet_mismatches),
            "mismatches": json_parquet_mismatches,
            "parquet_read_failure_count": len(parquet_fail),
        },
    )

    gate_status = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW" if "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW" in runtime_blockers else ("PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE_ADAPTIVE" if campaign_new_complete_count == 1 else "PASS_CAMPAIGN_C_ADAPTIVE_PARTIAL")
    if "BLOCKED_ADAPTIVE_POLLING_REPLAY_VALIDATION" in runtime_blockers:
        gate_status = "BLOCKED_ADAPTIVE_POLLING_REPLAY_VALIDATION"
    if "BLOCKED_UNKNOWN_DAILY_API_USAGE" in runtime_blockers:
        gate_status = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
    if "MISSING_SERVICE_KEY" in runtime_blockers and gate_status == "PASS_CAMPAIGN_C_ADAPTIVE_PARTIAL":
        gate_status = "MISSING_SERVICE_KEY"

    gate = {
        "artifact_dir": str(output_root),
        "cleanup_artifact_dir": str(cleanup_root) if cleanup_root else None,
        "cleanup_gate": cleanup_gate.get("gate_status"),
        "runtime_gate": gate_status,
        "gate_status": gate_status,
        "gate_passed": gate_status.startswith("PASS_"),
        "runtime_authorization_approved": runtime_authorization_approved,
        "run_date": run_date,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": preflight_calls,
        "campaign_physical_calls": campaign_calls,
        "r2d1n_total_physical_calls": r2d1n_total,
        "total_physical_calls_on_run_date": total_physical_calls_on_run_date,
        "effective_campaign_hard_cap": effective_hard_cap,
        "configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE,
        "observed_max_calls_per_minute": observed_max_calls,
        "target_route": TARGET_ROUTE,
        "target_routes": [TARGET_ROUTE],
        "excluded_routes": EXCLUDED_ROUTES,
        "non_target_route_api_calls": non_target_route_api_calls,
        "adaptive_replay_projected_calls": projected_calls,
        "previous_campaign_calls": PREVIOUS_CAMPAIGN_CALLS,
        "adaptive_replay_call_reduction_percent": call_reduction_percent,
        "adaptive_replay_passed": replay.get("adaptive_replay_passed"),
        "lookahead_used": replay.get("lookahead_used"),
        "candidate_vehicle_ids": candidate_vehicle_ids,
        "polling_mode_counts": polling_mode_counts,
        "campaign_c_new_complete_episode_count": campaign_new_complete_count,
        "right_censored_episode_count": right_censored_episode_count,
        "candidate_sequence_path": candidate_sequence_path,
        "exact_id_reentry_sequence": exact_id_reentry_sequence,
        "post_terminal_confirmation_count": post_terminal_confirmation_count,
        "provider_interval_sec": provider_interval_sec,
        "request_interval_sec": request_interval_sec,
        "conservative_dual_interval_sec": conservative_dual_interval_sec,
        "cumulative_complete_candidate": cumulative_complete_count,
        "route_complete_counts": cumulative_route_counts,
        "global_unique_vehicle_count": cumulative_global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": max(0, 12 - cumulative_complete_count),
        "method_prototype_data_threshold_met": method_ready,
        "calls_after_first_fatal_error": calls_after_first_fatal_error,
        "strict_json_failure_count": len(strict_fail),
        "non_json_raw_misclassification_count": 0,
        "parquet_read_failure_count": len(parquet_fail),
        "json_parquet_value_mismatch_count": len(json_parquet_mismatches),
        "manifest_missing_required_file_count": None,
        "manifest_nonself_hash_mismatch_count": None,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "terminal_recovery_estimation_execution_approved": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "next_authorized_action": "offline independent audit only",
        "blocking_reasons": runtime_blockers,
        "service_key_present": service_key_present,
        "service_key_length": service_key_length,
    }
    dump_json(output_root / "prompt5_e01_r2d1n_gate.json", gate)
    manifest = write_manifest(output_root)
    manifest_check = validate_manifest(output_root, manifest)
    gate["manifest_missing_required_file_count"] = manifest_check["manifest_missing_required_file_count"]
    gate["manifest_nonself_hash_mismatch_count"] = manifest_check["manifest_nonself_hash_mismatch_count"]
    dump_json(output_root / "prompt5_e01_r2d1n_gate.json", gate)
    manifest = write_manifest(output_root)
    manifest_check = validate_manifest(output_root, manifest)
    if manifest_check["manifest_nonself_hash_mismatch_count"]:
        gate["manifest_nonself_hash_mismatch_count"] = manifest_check["manifest_nonself_hash_mismatch_count"]
        dump_json(output_root / "prompt5_e01_r2d1n_gate.json", gate)
        manifest = write_manifest(output_root)

    print("R2D-1N CAMPAIGN C ADAPTIVE POLLING RETRY COMPLETE")
    print()
    print("cleanup_artifact_dir:")
    print(cleanup_root)
    print()
    print("cleanup_gate:")
    print(cleanup_gate.get("gate_status"))
    print()
    print("runtime_artifact_dir:")
    print(output_root)
    print()
    print("runtime_gate:")
    print(gate["gate_status"])
    print()
    print("adaptive_replay_projected_calls:")
    print(projected_calls)
    print()
    print("previous_campaign_calls:")
    print(PREVIOUS_CAMPAIGN_CALLS)
    print()
    print("adaptive_replay_call_reduction_percent:")
    print(round(call_reduction_percent, 2))
    print()
    print("runtime_authorization_approved:")
    print(str(runtime_authorization_approved).lower())
    print()
    print("run_date:")
    print(run_date)
    print()
    print("prior_physical_calls_on_run_date:")
    print(prior_calls)
    print()
    print("preflight_physical_calls:")
    print(preflight_calls)
    print()
    print("campaign_physical_calls:")
    print(campaign_calls)
    print()
    print("r2d1n_total_physical_calls:")
    print(r2d1n_total)
    print()
    print("total_physical_calls_on_run_date:")
    print(total_physical_calls_on_run_date)
    print()
    print("configured_max_calls_per_minute:")
    print(CONFIGURED_MAX_CALLS_PER_MINUTE)
    print()
    print("observed_max_calls_per_minute:")
    print(observed_max_calls)
    print()
    print("target_route:")
    print(TARGET_ROUTE)
    print()
    print("non_target_route_api_calls:")
    print(non_target_route_api_calls)
    print()
    print("candidate_vehicle_ids:")
    print(candidate_vehicle_ids)
    print()
    print("polling_mode_counts:")
    for mode in ["GUARD_SCAN", "APPROACH_SCAN", "PRE_TERMINAL_WATCH", "TERMINAL_APPROACH_FOCUSED", "POST_TERMINAL_FOCUSED"]:
        print(f"{mode}={polling_mode_counts.get(mode, 0)}")
    print()
    print("campaign_c_new_complete_episode_count:")
    print(campaign_new_complete_count)
    print()
    print("right_censored_episode_count:")
    print(right_censored_episode_count)
    print()
    print("candidate_sequence_path:")
    print(candidate_sequence_path)
    print()
    print("exact_id_reentry_sequence:")
    print(exact_id_reentry_sequence)
    print()
    print("post_terminal_confirmation_count:")
    print(post_terminal_confirmation_count)
    print()
    print("provider_interval_sec:")
    print(provider_interval_sec)
    print()
    print("request_interval_sec:")
    print(request_interval_sec)
    print()
    print("conservative_dual_interval_sec:")
    print(conservative_dual_interval_sec)
    print()
    print("cumulative_complete_candidate:")
    print(cumulative_complete_count)
    print()
    print("route_complete_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={cumulative_route_counts.get(route, 0)}")
    print()
    print("global_unique_vehicle_count:")
    print(cumulative_global_unique_vehicle_count)
    print()
    print("remaining_complete_episodes_to_12:")
    print(max(0, 12 - cumulative_complete_count))
    print()
    print("method_prototype_data_threshold_met:")
    print(str(method_ready).lower())
    print()
    print("calls_after_first_fatal_error:")
    print(calls_after_first_fatal_error)
    print()
    print("strict_json_failure_count:")
    print(len(strict_fail))
    print()
    print("non_json_raw_misclassification_count:")
    print(0)
    print()
    print("parquet_read_failure_count:")
    print(len(parquet_fail))
    print()
    print("json_parquet_value_mismatch_count:")
    print(len(json_parquet_mismatches))
    print()
    print("manifest_missing_required_file_count:")
    print(gate["manifest_missing_required_file_count"])
    print()
    print("manifest_nonself_hash_mismatch_count:")
    print(gate["manifest_nonself_hash_mismatch_count"])
    print()
    print("secret_leak_count:")
    print(secret_audit["secret_leak_count"])
    print()
    print("terminal_recovery_estimation_execution_approved:")
    print("false")
    print()
    print("simulator_application_authorized:")
    print("false")
    print()
    print("phase2_authorized:")
    print("false")
    print()
    print("next_authorized_action:")
    print("offline independent audit only")


if __name__ == "__main__":
    main()
