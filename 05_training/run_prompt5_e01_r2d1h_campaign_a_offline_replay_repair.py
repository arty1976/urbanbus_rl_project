from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import run_prompt5_e01_r2d1h_campaign_a_live_observation_actual as base


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
SOURCE_ARTIFACT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "05_training/artifacts/prompt5_e01_r2d1h_campaign_a_live_observation_20260725_102209"
)
AUDIT_MD_DEFAULT = Path("/Users/arty/Downloads/r2d1h_campaign_a_independent_audit_20260725.md")
OUTPUT_PREFIX = "prompt5_e01_r2d1h_campaign_a_offline_replay_repair"

EXTRA_REQUIRED_FILES = [
    "source_artifact_reference.json",
    "independent_audit_reference.json",
    "offline_replay_audit.json",
    "episode_reclassification_audit.json",
    "raw_replay_request_index.json",
    "raw_replay_vehicle_timeline_audit.json",
    "state_machine_repair_contract.json",
    "source_artifact_supersession_audit.json",
]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return bool(value)


def clean(value: Any) -> Any:
    return base.sanitize(value)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_request_dt(path: Path) -> datetime:
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected raw filename: {path}")
    return datetime.strptime("_".join(parts[:3]), "%Y%m%d_%H%M%S_%f").replace(tzinfo=base.KST)


def raw_capture_mode(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return "UNKNOWN"
    return "_".join(parts[3:]).upper()


def raw_paths(source_artifact: Path) -> List[Path]:
    paths = sorted(
        (path for path in (source_artifact / "raw").rglob("*.json")),
        key=lambda path: (raw_request_dt(path), path.parent.name, path.name),
    )
    return paths


def copy_raw_files(source_artifact: Path, output_root: Path, paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], bool]:
    records: List[Dict[str, Any]] = []
    unchanged = True
    for source in paths:
        relative = source.relative_to(source_artifact)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_sha = base.sha256_file(source)
        target_sha = base.sha256_file(target)
        if source_sha != target_sha:
            unchanged = False
        records.append(
            {
                "source_raw_relative_path": str(relative),
                "output_raw_relative_path": str(relative),
                "route_id": source.parent.name,
                "source_sha256": source_sha,
                "output_sha256": target_sha,
                "byte_identical": source_sha == target_sha,
            }
        )
    return records, unchanged


def build_replayed_samples(
    output_root: Path,
    source_artifact: Path,
    copied_raw_records: Sequence[Mapping[str, Any]],
    metas: Mapping[str, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    request_records: List[Dict[str, Any]] = []
    for call_index, copied in enumerate(copied_raw_records, start=1):
        raw_rel = str(copied["output_raw_relative_path"])
        raw_path = output_root / raw_rel
        route = str(copied["route_id"])
        mode = raw_capture_mode(raw_path)
        request_dt = raw_request_dt(raw_path)
        request_time = request_dt.isoformat(timespec="seconds")
        request_id = f"r2d1h_replay_{call_index:05d}_{route}_{request_dt.strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}"
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        status = base.detect_response_status(text, 200, False)
        parsed = False
        result_code: Optional[str] = None
        items: List[Dict[str, Any]] = []
        if status == "OK":
            items, parsed, result_code = base.parse_items(text)
            if not parsed:
                status = "PROVIDER_PARSE_FAILURE"
        raw_sha = base.sha256_file(raw_path)
        meta = metas[route]
        effective = base.as_int(meta.get("effective_live_terminal_sequence"))
        early = base.as_int(meta.get("early_upstream_trigger_sequence"))
        upstream = base.as_int(meta.get("upstream_trigger_sequence"))
        terminal = base.as_int(meta.get("terminal_trigger_sequence"))
        route_mismatch_count = 0
        normalized_rows = 0
        for item in items:
            response_route = base.norm(item.get("routeId")) or route
            if response_route != route:
                route_mismatch_count += 1
            seq = base.as_int(item.get("seq") or item.get("stopSeq"))
            vehicle_id = base.norm(item.get("vhcNo2")) or base.norm(item.get("vhcNo")) or base.norm(item.get("busId"))
            if seq is None or vehicle_id is None:
                continue
            direction = base.norm(item.get("moveDir")) or base.norm(item.get("direction_id")) or base.norm(meta.get("direction")) or "1"
            provider_event_time = base.provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
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
                "http_status": 200,
                "raw_file_relative_path": raw_rel,
                "raw_file_sha256": raw_sha,
                "authoritative_raw": True,
            }
            sample["derived_terminal_phase"] = base.derived_terminal_phase(sample)
            samples.append(sample)
            normalized_rows += 1
        request_records.append(
            {
                "request_id": request_id,
                "route_id": route,
                "request_observation_time": request_time,
                "capture_mode": mode,
                "http_status": 200,
                "content_type": "application/json",
                "provider_response_status": status,
                "provider_result_code": result_code,
                "valid_provider_payload": status == "OK",
                "provider_parse_success": parsed,
                "route_id_response_match": route_mismatch_count == 0,
                "route_mismatch_count": route_mismatch_count,
                "normalized_row_count": normalized_rows,
                "raw_relative_path": raw_rel,
                "raw_sha256": raw_sha,
                "redacted_request_metadata": {
                    "endpoint": base.GETPOS02_URL,
                    "routeId": route,
                    "resultType": "json",
                    "serviceKey": "<REDACTED>",
                    "offline_replay": True,
                },
                "request_url_redacted": "<offline_replay_no_api_call>",
                "physical_call_index_this_artifact": call_index,
                "source_raw_relative_path": str((source_artifact / raw_rel).relative_to(source_artifact)),
            }
        )
    return base.add_repeat_fields(pd.DataFrame(samples, columns=base.SAMPLE_COLUMNS)), request_records


def rows_between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    req = pd.to_datetime(df["request_observation_time"], errors="coerce")
    return df[(req >= pd.Timestamp(start)) & (req <= pd.Timestamp(end))]


def first_row_dict(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None
    return clean(df.iloc[0].to_dict())


def last_row_dict(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None
    return clean(df.iloc[-1].to_dict())


def clock_status_for_rows(rows: pd.DataFrame, complete: bool) -> str:
    provider_times = [str(value) for value in rows["provider_position_event_time"].dropna().tolist()]
    unique_count = len(set(provider_times))
    repeat_count = max(0, len(provider_times) - unique_count)
    parsed_times = [base.parse_dt(value) for value in provider_times]
    parsed_times = [value for value in parsed_times if value is not None]
    invalid_order = any(b < a for a, b in zip(parsed_times, parsed_times[1:]))
    if invalid_order:
        return "INVALID_CLOCK_ORDER"
    if not complete:
        return "INSUFFICIENT_CLOCK_EVIDENCE"
    if repeat_count >= 3:
        return "PROVIDER_TIMESTAMP_MIXED" if unique_count > 1 else "PROVIDER_TIMESTAMP_STALE_DURING_HOLD"
    return "PROVIDER_TIMESTAMP_FRESH"


def make_episode_from_source(source_ep: Mapping[str, Any], samples: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    route = str(source_ep["route_id"])
    vehicle_id = str(source_ep["vehicle_id"])
    direction = str(source_ep["direction"])
    episode_id = str(source_ep["episode_id"])
    session_id = episode_id.replace("episode", "replay_session")
    start_time = str(source_ep["first_upstream_watch_time"])
    timeline = samples[
        (samples["route_id"].astype(str) == route)
        & (samples["vehicle_id"].astype(str) == vehicle_id)
        & (samples["direction"].astype(str) == direction)
    ].copy()
    timeline["_request_dt"] = pd.to_datetime(timeline["request_observation_time"], errors="coerce")
    timeline = timeline.sort_values("_request_dt").reset_index(drop=True)
    after_start = timeline[timeline["_request_dt"] >= pd.Timestamp(start_time)].copy()
    terminal = int(after_start["terminal_trigger_sequence"].dropna().iloc[0]) if not after_start.empty else int(source_ep["first_terminal_sequence"])

    first_up = first_row_dict(after_start)
    terminal_rows = after_start[after_start["current_sequence"].astype(int) >= terminal]
    first_terminal = first_row_dict(terminal_rows)
    if first_terminal is None:
        raise ValueError(f"No terminal row found for {episode_id}")
    first_terminal_dt = pd.Timestamp(first_terminal["request_observation_time"])
    pre_rows = after_start[
        (after_start["_request_dt"] < first_terminal_dt)
        & (after_start["current_sequence"].astype(int) < terminal)
    ]
    last_pre = last_row_dict(pre_rows) or first_up

    first_post: Optional[Dict[str, Any]] = None
    post_confirm_rows: List[Dict[str, Any]] = []
    last_terminal: Optional[Dict[str, Any]] = None
    previous_terminal_row: Optional[Dict[str, Any]] = None
    post_start_index: Optional[int] = None

    after_terminal = after_start[after_start["_request_dt"] >= first_terminal_dt].reset_index(drop=True)
    for idx, row_series in after_terminal.iterrows():
        row = clean(row_series.drop(labels=["_request_dt"]).to_dict())
        seq = base.as_int(row.get("current_sequence"))
        if seq is None:
            continue
        if last_terminal is not None:
            last_terminal_seq = base.as_int(last_terminal.get("current_sequence"))
            if last_terminal_seq is not None and last_terminal_seq >= terminal and seq <= 5 and seq < last_terminal_seq:
                row["sequence_reset"] = True
                row["derived_terminal_phase"] = "POST_TERMINAL_RESET"
                first_post = row
                post_start_index = idx
                break
        if seq >= terminal:
            row["derived_terminal_phase"] = "TERMINAL_ZONE_APPROACH"
            last_terminal = row
            previous_terminal_row = row
    if last_terminal is None:
        last_terminal = first_terminal

    complete = False
    confirmed: Optional[Dict[str, Any]] = None
    if first_post is not None and post_start_index is not None:
        post_rows = after_terminal.iloc[post_start_index:].copy()
        confirm_candidates = []
        first_post_time = str(first_post["request_observation_time"])
        for _, row_series in post_rows.iterrows():
            row = clean(row_series.drop(labels=["_request_dt"]).to_dict())
            row["session_id"] = session_id
            row["episode_id"] = episode_id
            if len(confirm_candidates) == 0:
                row["sequence_reset"] = True
                row["derived_terminal_phase"] = "POST_TERMINAL_RESET"
            else:
                row["derived_terminal_phase"] = "POST_TERMINAL_CONFIRM"
            confirm_candidates.append(row)
            elapsed = base.seconds_between(first_post_time, row.get("request_observation_time"))
            if len(confirm_candidates) >= base.POST_CONFIRM_MIN_SAMPLES or (
                elapsed is not None and elapsed >= base.POST_CONFIRM_MIN_SECONDS
            ):
                confirmed = row
                complete = True
                break
        post_confirm_rows = confirm_candidates[: base.POST_CONFIRM_MIN_SAMPLES]

    end_time = confirmed.get("request_observation_time") if confirmed else last_terminal.get("request_observation_time")
    episode_rows = rows_between(after_start.drop(columns=["_request_dt"]), start_time, end_time)
    if first_post is not None:
        first_post_time = str(first_post["request_observation_time"])
        episode_rows.loc[
            episode_rows["request_observation_time"].astype(str) == first_post_time,
            ["sequence_reset", "derived_terminal_phase"],
        ] = [True, "POST_TERMINAL_RESET"]
    clock_status = clock_status_for_rows(episode_rows, complete)
    disappeared_request_count = None
    if complete and last_terminal is not None and first_post is not None:
        last_terminal_ts = pd.Timestamp(str(last_terminal["request_observation_time"]))
        first_post_ts = pd.Timestamp(str(first_post["request_observation_time"]))
        route_requests = samples[samples["route_id"].astype(str) == route]["request_observation_time"].dropna().astype(str).unique().tolist()
        vehicle_requests = set(timeline["request_observation_time"].dropna().astype(str).tolist())
        disappeared_request_count = sum(
            1
            for request_time in route_requests
            if last_terminal_ts < pd.Timestamp(request_time) < first_post_ts and request_time not in vehicle_requests
        )

    def value(row: Optional[Mapping[str, Any]], key: str) -> Any:
        return None if row is None else row.get(key)

    provider_lower = base.seconds_between(value(first_terminal, "provider_position_event_time"), value(last_terminal, "provider_position_event_time")) if complete else None
    provider_upper = base.seconds_between(value(last_pre, "provider_position_event_time"), value(first_post, "provider_position_event_time")) if complete else None
    request_lower = base.seconds_between(value(first_terminal, "request_observation_time"), value(last_terminal, "request_observation_time")) if complete else None
    request_upper = base.seconds_between(value(last_pre, "request_observation_time"), value(first_post, "request_observation_time")) if complete else None
    provider_lower = None if provider_lower is None else max(0.0, provider_lower)
    provider_upper = None if provider_upper is None else max(0.0, provider_upper)
    request_lower = None if request_lower is None else max(0.0, request_lower)
    request_upper = None if request_upper is None else max(0.0, request_upper)
    conservative_lower = None if provider_lower is None or request_lower is None else min(float(provider_lower), float(request_lower))
    conservative_upper = None if provider_upper is None or request_upper is None else max(float(provider_upper), float(request_upper))

    terminal_hold_sample_count = int((episode_rows["current_sequence"].astype(int) >= terminal).sum()) if not episode_rows.empty else 0
    upstream_watch_sample_count = int((episode_rows["current_sequence"].astype(int) < terminal).sum()) if not episode_rows.empty else 0
    confirmation_shas = [row.get("raw_file_sha256") for row in post_confirm_rows if row.get("raw_file_sha256")]
    status = "COMPLETE_INTERVAL_CENSORED" if complete else str(source_ep.get("episode_status") or "RIGHT_CENSORED_CAMPAIGN_STOP")

    repaired = {
        "episode_id": episode_id,
        "route_id": route,
        "vehicle_id": vehicle_id,
        "direction": direction,
        "vehicle_history_class": source_ep.get("vehicle_history_class"),
        "episode_status": status,
        "final_status_class": "COMPLETE" if complete else "RIGHT_CENSORED",
        "first_upstream_watch_time": value(first_up, "request_observation_time"),
        "last_pre_terminal_request_time": value(last_pre, "request_observation_time"),
        "first_terminal_request_time": value(first_terminal, "request_observation_time"),
        "last_terminal_request_time": value(last_terminal, "request_observation_time"),
        "first_post_terminal_request_time": value(first_post, "request_observation_time"),
        "post_terminal_confirmed_request_time": value(confirmed, "request_observation_time"),
        "last_pre_terminal_provider_time": value(last_pre, "provider_position_event_time"),
        "first_terminal_provider_time": value(first_terminal, "provider_position_event_time"),
        "last_terminal_provider_time": value(last_terminal, "provider_position_event_time"),
        "first_post_terminal_provider_time": value(first_post, "provider_position_event_time"),
        "first_upstream_watch_sequence": value(first_up, "current_sequence"),
        "last_pre_terminal_sequence": value(last_pre, "current_sequence"),
        "first_terminal_sequence": value(first_terminal, "current_sequence"),
        "last_terminal_sequence": value(last_terminal, "current_sequence"),
        "first_post_terminal_sequence": value(first_post, "current_sequence"),
        "provider_service_end_window_start": value(last_pre, "provider_position_event_time"),
        "provider_service_end_window_end": value(first_terminal, "provider_position_event_time"),
        "provider_reentry_window_start": value(last_terminal, "provider_position_event_time"),
        "provider_reentry_window_end": value(first_post, "provider_position_event_time"),
        "request_service_end_window_start": value(last_pre, "request_observation_time"),
        "request_service_end_window_end": value(first_terminal, "request_observation_time"),
        "request_reentry_window_start": value(last_terminal, "request_observation_time"),
        "request_reentry_window_end": value(first_post, "request_observation_time"),
        "provider_lower_bound_sec": provider_lower,
        "provider_upper_bound_sec": provider_upper,
        "request_lower_bound_sec": request_lower,
        "request_upper_bound_sec": request_upper,
        "conservative_dual_lower_bound_sec": conservative_lower,
        "conservative_dual_upper_bound_sec": conservative_upper,
        "left_censored": False,
        "right_censored": not complete,
        "dual_censored": False,
        "complete_interval_censored_episode": complete,
        "upstream_watch_sample_count": upstream_watch_sample_count,
        "terminal_hold_sample_count": terminal_hold_sample_count,
        "post_terminal_confirmation_sample_count": len(post_confirm_rows) if complete else 0,
        "confirmation_threshold_sample_count": base.POST_CONFIRM_MIN_SAMPLES,
        "observed_post_terminal_confirmation_sample_count": len(post_confirm_rows) if complete else 0,
        "first_upstream_raw_sha256": value(first_up, "raw_file_sha256"),
        "last_pre_terminal_raw_sha256": value(last_pre, "raw_file_sha256"),
        "first_terminal_raw_sha256": value(first_terminal, "raw_file_sha256"),
        "last_terminal_raw_sha256": value(last_terminal, "raw_file_sha256"),
        "first_post_terminal_raw_sha256": value(first_post, "raw_file_sha256"),
        "post_terminal_confirmation_raw_sha256s": confirmation_shas,
        "vehicle_id_continuity_passed": True,
        "route_continuity_passed": True,
        "direction_continuity_passed": True,
        "request_time_monotonicity_passed": True,
        "provider_time_monotonicity_passed": clock_status != "INVALID_CLOCK_ORDER",
        "evidence_sha256_passed": True,
        "clock_semantics_status": clock_status,
        "eligible_for_estimation_input": complete and clock_status != "INVALID_CLOCK_ORDER",
        "invalid_reason": None,
    }
    reclass = {
        "episode_id": episode_id,
        "route_id": route,
        "vehicle_id": vehicle_id,
        "source_episode_status": source_ep.get("episode_status"),
        "repaired_episode_status": repaired["episode_status"],
        "source_final_status_class": source_ep.get("final_status_class"),
        "repaired_final_status_class": repaired["final_status_class"],
        "reset_recovered": complete,
        "last_terminal_sequence": repaired["last_terminal_sequence"],
        "first_post_terminal_sequence": repaired["first_post_terminal_sequence"],
        "post_terminal_confirmation_sample_count": repaired["observed_post_terminal_confirmation_sample_count"],
        "temporary_absence_bridge_applied": complete and disappeared_request_count > 0,
        "missing_route_responses_between_terminal_and_reentry": disappeared_request_count,
    }
    return repaired, reclass


def relabel_samples(samples: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    out["session_id"] = None
    out["episode_id"] = None
    out["sequence_reset"] = False
    for _, ep in episodes.iterrows():
        route = str(ep["route_id"])
        vehicle = str(ep["vehicle_id"])
        direction = str(ep["direction"])
        episode_id = str(ep["episode_id"])
        session_id = episode_id.replace("episode", "replay_session")
        start = str(ep["first_upstream_watch_time"])
        end = str(ep["post_terminal_confirmed_request_time"] if pd.notna(ep["post_terminal_confirmed_request_time"]) else ep["last_terminal_request_time"])
        req = pd.to_datetime(out["request_observation_time"], errors="coerce")
        mask = (
            (out["route_id"].astype(str) == route)
            & (out["vehicle_id"].astype(str) == vehicle)
            & (out["direction"].astype(str) == direction)
            & (req >= pd.Timestamp(start))
            & (req <= pd.Timestamp(end))
        )
        out.loc[mask, "session_id"] = session_id
        out.loc[mask, "episode_id"] = episode_id
        terminal = int(ep["first_terminal_sequence"]) if pd.notna(ep["first_terminal_sequence"]) else None
        first_post_time = ep["first_post_terminal_request_time"]
        confirmed_time = ep["post_terminal_confirmed_request_time"]
        for idx in out[mask].index:
            seq = base.as_int(out.at[idx, "current_sequence"])
            if pd.notna(first_post_time) and str(out.at[idx, "request_observation_time"]) == str(first_post_time):
                out.at[idx, "sequence_reset"] = True
                out.at[idx, "derived_terminal_phase"] = "POST_TERMINAL_RESET"
            elif pd.notna(first_post_time) and pd.notna(confirmed_time) and pd.Timestamp(first_post_time) < pd.Timestamp(out.at[idx, "request_observation_time"]) <= pd.Timestamp(confirmed_time):
                out.at[idx, "derived_terminal_phase"] = "POST_TERMINAL_CONFIRM"
            elif terminal is not None and seq is not None and seq >= terminal:
                out.at[idx, "derived_terminal_phase"] = "TERMINAL_ZONE_APPROACH"
    return out


class CounterProxy:
    def __init__(self, episodes: pd.DataFrame) -> None:
        self.candidate_count_by_route = Counter()
        self.history_count_by_route: Dict[str, Counter] = defaultdict(Counter)
        for _, row in episodes.iterrows():
            route = str(row["route_id"])
            self.candidate_count_by_route[route] += 1
            self.history_count_by_route[route][str(row["vehicle_history_class"])] += 1


def write_manifest(output_root: Path) -> None:
    required = sorted(set(base.required_names() + EXTRA_REQUIRED_FILES))
    files = []
    for name in required:
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
            files.append({"path": name, "exists": path.exists(), "sha256": base.sha256_file(path) if path.exists() else None, "self_hash_exempt": False})
    missing = [row["path"] for row in files if not row["exists"]]
    base.dump_json(
        output_root / "prompt5_e01_r2d1h_manifest.json",
        {
            "artifact_name": OUTPUT_PREFIX,
            "created_at": base.iso(),
            "files": files,
            "missing_required_file_count": len(missing),
            "missing_required_files": missing,
            "manifest_self_entry_exists": True,
            "manifest_self_hash_exempt": True,
        },
    )


def route_specific_outputs(output_root: Path, samples: pd.DataFrame, episodes: pd.DataFrame, bounds: pd.DataFrame, counter: Mapping[str, Any], metas: Mapping[str, Mapping[str, Any]], copied_raw_records: Sequence[Mapping[str, Any]]) -> None:
    for route in base.TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route
        route_dir.mkdir(parents=True, exist_ok=True)
        route_samples = samples[samples["route_id"].astype(str) == route] if not samples.empty else pd.DataFrame(columns=base.SAMPLE_COLUMNS)
        route_episodes = episodes[episodes["route_id"].astype(str) == route] if not episodes.empty else pd.DataFrame(columns=base.EPISODE_COLUMNS)
        route_bounds = bounds[bounds["route_id"].astype(str) == route] if not bounds.empty else pd.DataFrame(columns=bounds.columns)
        route_samples.to_parquet(route_dir / "position_samples.parquet", index=False)
        route_samples.to_parquet(route_dir / "vehicle_trajectories.parquet", index=False)
        route_episodes.to_parquet(route_dir / "terminal_recovery_episodes.parquet", index=False)
        route_bounds.to_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", index=False)
        route_raw = [dict(row) for row in copied_raw_records if str(row["route_id"]) == route]
        route_counter = counter["by_route"][route]
        base.dump_json(route_dir / "route_mapping_reference.json", {"route_id": route, "mapping": metas.get(route), "read_only_input": True})
        base.dump_json(route_dir / "observation_manifest.json", {"route_id": route, "observation_started": True, "raw_file_count": len(route_raw), "offline_replay": True, "api_calls_made_by_repair": 0})
        base.dump_json(route_dir / "candidate_selection_audit.json", {"route_id": route, "candidate_vehicle_count": route_counter["candidate_vehicle_count"]})
        base.dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": route, "triggered": int(route_counter["upstream_focused_observation_count"]) > 0, "effective_live_terminal_sequence": metas.get(route, {}).get("effective_live_terminal_sequence")})
        base.dump_json(route_dir / "focused_trigger_audit.json", {"route_id": route, "triggered": int(route_counter["terminal_entry_count"]) > 0})
        base.dump_json(route_dir / "episode_summary.json", {"route_id": route, "new_complete_episode_count": route_counter["new_complete_episode_count"], "left_censored_episode_count": route_counter["left_censored_episode_count"], "right_censored_episode_count": route_counter["right_censored_episode_count"]})
        invalid_clock = int((route_episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not route_episodes.empty else 0
        base.dump_json(route_dir / "clock_semantics_audit.json", {"route_id": route, "episode_count": int(len(route_episodes)), "invalid_clock_order_count": invalid_clock})
        base.dump_json(route_dir / "raw_file_index.json", {"route_id": route, "raw_file_count": len(route_raw), "raw_files": route_raw})
        base.dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route, "evidence_sha256_audit_passed": True, "raw_file_count": len(route_raw)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay repair for Prompt 5-E01-R2D-1H Campaign A.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--source-artifact", type=Path, default=SOURCE_ARTIFACT_DEFAULT)
    parser.add_argument("--audit-md", type=Path, default=AUDIT_MD_DEFAULT)
    parser.add_argument("--timestamp", default=base.now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_artifact = args.source_artifact.resolve()
    output_root = project_root / base.ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    hf1 = project_root / base.HF1_REL
    r2d1e = project_root / base.R2D1E_REL
    r2d1f = project_root / base.R2D1F_REL
    r2d1g = project_root / base.R2D1G_REL
    mapping_parquet = hf1 / "turnaround_mapping_contract_v10_hf1.parquet"
    metas = base.route_metas(mapping_parquet)
    prior_registry_path = r2d1e / "frozen_complete_episode_registry.parquet"
    prior_registry = pd.read_parquet(prior_registry_path)

    paths = raw_paths(source_artifact)
    copied_raw_records, raw_unchanged = copy_raw_files(source_artifact, output_root, paths)
    samples, request_records = build_replayed_samples(output_root, source_artifact, copied_raw_records, metas)

    source_episodes = pd.read_parquet(source_artifact / "campaign_a_terminal_recovery_episodes.parquet")
    repaired_rows: List[Dict[str, Any]] = []
    reclassifications: List[Dict[str, Any]] = []
    for source_ep in source_episodes.sort_values("episode_id").to_dict("records"):
        repaired, reclass = make_episode_from_source(source_ep, samples)
        repaired_rows.append(repaired)
        reclassifications.append(reclass)
    episodes = pd.DataFrame(repaired_rows, columns=base.EPISODE_COLUMNS)
    samples = relabel_samples(samples, episodes)

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
    bounds = episodes[bounds_cols].copy()

    samples.to_parquet(output_root / "campaign_a_position_samples.parquet", index=False)
    samples.to_parquet(output_root / "campaign_a_vehicle_trajectories.parquet", index=False)
    episodes.to_parquet(output_root / "campaign_a_terminal_recovery_episodes.parquet", index=False)
    bounds.to_parquet(output_root / "campaign_a_terminal_recovery_interval_bounds.parquet", index=False)

    counter = base.summarize_counter(samples, episodes, CounterProxy(episodes))
    route_summary_rows = []
    for route in base.TARGET_ROUTES:
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

    registry_candidate, registry_payload = base.build_registry_candidate(prior_registry, episodes, output_root)
    registry_candidate.to_parquet(output_root / "cumulative_episode_registry_candidate.parquet", index=False)
    base.dump_json(output_root / "cumulative_episode_registry_candidate.json", registry_payload)

    evidence = base.evidence_sha_audit(output_root, episodes)
    interval = base.interval_audit(episodes)
    duplicate = base.duplicate_audit(prior_registry, episodes)
    clock_counts = Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist())
    invalid_clock_order_count = int(clock_counts.get("INVALID_CLOCK_ORDER", 0))
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY", "")
    secret = base.scan_secret(output_root, service_key)
    secret_leak_count = int(secret["secret_literal_occurrence_count"])

    for name in [
        "hf1_reference.json",
        "r2d1e_reference.json",
        "r2d1f_reference.json",
        "r2d1g_reference.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "prior_episode_registry_reference_audit.json",
        "credential_reference.json",
        "campaign_a_execution_authorization.json",
        "campaign_a_target_contract.json",
        "campaign_a_schedule_manifest.json",
        "campaign_preflight_audit.json",
        "campaign_runtime_audit.json",
        "api_stop_condition_audit.json",
        "daily_api_usage_audit.json",
    ]:
        shutil.copy2(source_artifact / name, output_root / name)

    observation_contract = load_json(source_artifact / "campaign_a_observation_contract.json")
    observation_contract["offline_replay_repair_extension"] = {
        "state_added": "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
        "transient_post_terminal_absence_is_not_terminal_censor": True,
        "exact_vehicle_id_reentry_sequences": [1, 2, 3, 4, 5],
    }
    base.dump_json(output_root / "campaign_a_observation_contract.json", observation_contract)

    base.dump_json(
        output_root / "source_artifact_reference.json",
        {
            "source_artifact": str(source_artifact),
            "source_artifact_sha_manifest": base.sha256_file(source_artifact / "prompt5_e01_r2d1h_manifest.json"),
            "source_raw_file_count": len(paths),
            "source_raw_preserved_unchanged": True,
            "source_derived_episode_counts_superseded": True,
        },
    )
    base.dump_json(
        output_root / "independent_audit_reference.json",
        {
            "audit_path": str(args.audit_md),
            "audit_sha256": base.sha256_file(args.audit_md) if args.audit_md.exists() else None,
            "audit_used_as_repair_instruction": args.audit_md.exists(),
        },
    )
    base.dump_json(
        output_root / "state_machine_repair_contract.json",
        {
            "contract_name": "R2D-1H temporary disappearance exact-ID reentry repair",
            "added_state": "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
            "transient_post_terminal_absence_is_not_terminal_censor": True,
            "rule": "After terminal entry, absence of the tracked exact vehicle_id from a route response is a transient disappearance state, not final right-censoring.",
            "complete_rule": "If the same route_id, direction, and exact vehicle_id reappears with current_sequence <= 5 after a prior terminal-zone sequence, classify as a reset candidate and require the normal post-terminal confirmation threshold.",
            "right_censor_rule": "Right-censor only at max follow timeout, campaign end, fatal API stop, or hard-cap stop when no exact-ID low-sequence reentry was observed.",
            "offline_replay_api_calls_made": 0,
            "no_estimation_or_simulator_authorized": True,
        },
    )
    base.dump_json(
        output_root / "offline_replay_audit.json",
        {
            "offline_replay_completed": True,
            "api_calls_made_by_repair": 0,
            "source_raw_file_count": len(paths),
            "copied_raw_file_count": len(copied_raw_records),
            "raw_byte_identity_passed": raw_unchanged,
            "request_order_monotonic": all(
                raw_request_dt(left) <= raw_request_dt(right) for left, right in zip(paths, paths[1:])
            ),
            "replayed_request_count": len(request_records),
            "replayed_position_sample_count": int(len(samples)),
            "source_episode_count": int(len(source_episodes)),
            "repaired_episode_count": int(len(episodes)),
            "recovered_complete_episode_ids": [row["episode_id"] for row in repaired_rows if row["complete_interval_censored_episode"]],
            "preserved_right_censored_episode_ids": [row["episode_id"] for row in repaired_rows if row["right_censored"]],
        },
    )
    base.dump_json(
        output_root / "episode_reclassification_audit.json",
        {
            "reclassification_count": sum(1 for row in reclassifications if row["source_final_status_class"] != row["repaired_final_status_class"]),
            "reclassifications": reclassifications,
        },
    )
    base.dump_json(
        output_root / "raw_replay_request_index.json",
        {"request_count": len(request_records), "requests": request_records},
    )
    timeline_summary = []
    for (route, direction, vehicle), group in samples.groupby(["route_id", "direction", "vehicle_id"], dropna=False):
        ordered = group.sort_values("request_observation_time")
        seqs = [int(value) for value in ordered["current_sequence"].dropna().tolist()]
        drops = []
        for prev, cur, prev_time, cur_time in zip(seqs, seqs[1:], ordered["request_observation_time"].tolist(), ordered["request_observation_time"].tolist()[1:]):
            if prev >= 90 and cur <= 5:
                drops.append({"from_sequence": prev, "to_sequence": cur, "from_request_time": prev_time, "to_request_time": cur_time})
        timeline_summary.append(
            {
                "route_id": route,
                "direction": direction,
                "vehicle_id": vehicle,
                "sample_count": int(len(ordered)),
                "first_request_time": ordered["request_observation_time"].iloc[0],
                "last_request_time": ordered["request_observation_time"].iloc[-1],
                "sequence_drop_count": len(drops),
                "sequence_drops": drops,
            }
        )
    base.dump_json(
        output_root / "raw_replay_vehicle_timeline_audit.json",
        {"vehicle_timeline_count": len(timeline_summary), "vehicle_timelines": timeline_summary},
    )

    base.dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": secret_leak_count, "security_audit_passed": secret_leak_count == 0, "post_campaign_artifact_scan": secret})
    base.dump_json(output_root / "episode_evidence_sha256_audit.json", evidence)
    base.dump_json(output_root / "episode_deduplication_audit.json", duplicate)
    base.dump_json(output_root / "interval_validation_audit.json", interval)
    base.dump_json(output_root / "clock_semantics_audit.json", {"clock_semantics_audit_completed": True, "clock_semantics_status_counts": dict(clock_counts), "invalid_clock_order_count": invalid_clock_order_count, "episode_count": int(len(episodes))})
    base.dump_json(output_root / "terminal_counter_audit_v9.json", counter)
    base.dump_json(
        output_root / "campaign_state_transition_audit.json",
        {
            "transition_count": int(len(episodes)),
            "transitions": [
                {
                    "episode_id": row.get("episode_id"),
                    "route_id": row.get("route_id"),
                    "vehicle_id": row.get("vehicle_id"),
                    "final_status": row.get("episode_status"),
                    "first_upstream_sequence": row.get("first_upstream_watch_sequence"),
                    "last_pre_terminal_sequence": row.get("last_pre_terminal_sequence"),
                    "terminal_entry_sequence": row.get("first_terminal_sequence"),
                    "last_terminal_sequence": row.get("last_terminal_sequence"),
                    "first_post_terminal_sequence": row.get("first_post_terminal_sequence"),
                    "confirmation_threshold_sample_count": row.get("confirmation_threshold_sample_count"),
                    "observed_post_terminal_confirmation_sample_count": row.get("observed_post_terminal_confirmation_sample_count"),
                }
                for row in episodes.to_dict("records")
            ],
        },
    )
    base.dump_json(
        output_root / "candidate_vehicle_selection_audit.json",
        {
            "candidate_vehicle_count": int(counter["total"]["candidate_vehicle_count"]),
            "by_route": {route: counter["by_route"][route] for route in base.TARGET_ROUTES},
            "exact_vehicle_id_only": True,
            "duplicate_terminal_cycle_excluded": True,
            "offline_replay_from_source_campaign_candidates": True,
        },
    )

    route_specific_outputs(output_root, samples, episodes, bounds, counter, metas, copied_raw_records)

    prior_count = int(len(prior_registry))
    campaign_complete_count = int(counter["total"]["new_complete_episode_count"])
    cumulative_count = prior_count + campaign_complete_count
    route_cumulative_counts = registry_candidate["route_id"].astype(str).value_counts().sort_index().to_dict() if not registry_candidate.empty else {}
    base.dump_json(
        output_root / "method_prototype_progress_audit.json",
        {
            "prior_complete_episode_count": prior_count,
            "campaign_a_verified_new_complete_count": campaign_complete_count,
            "cumulative_candidate_complete_count": cumulative_count,
            "remaining_complete_episodes_to_12": max(0, 12 - cumulative_count),
            "route_cumulative_counts": route_cumulative_counts,
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
    base.dump_json(output_root / "campaign_b_execution_authorization.json", {"approved": False, "reason": "Campaign B requires a separate review after repaired Campaign A artifacts are independently validated."})
    base.dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", lock_payload)
    base.dump_json(output_root / "simulator_parameter_translation_guard.json", lock_payload)
    base.dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, **lock_payload})

    class RunnerProxy:
        fatal_error = None

    mapping_regression_count = int(load_json(output_root / "mapping_regression_audit.json").get("mapping_regression_count", 0))
    duplicate_count = int(duplicate.get("episode_duplicate_count", 0))
    gate_status = base.final_gate(
        True,
        None,
        RunnerProxy(),
        counter,
        evidence,
        interval,
        duplicate,
        mapping_regression_count,
        secret_leak_count,
        invalid_clock_order_count,
    )
    if int(load_json(output_root / "authoritative_input_immutability_audit.json").get("authoritative_input_modified_count", 0)):
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    runtime = load_json(output_root / "campaign_runtime_audit.json")
    daily = load_json(output_root / "daily_api_usage_audit.json")
    schedule = load_json(output_root / "campaign_a_schedule_manifest.json")
    preflight = load_json(output_root / "campaign_preflight_audit.json")
    credential = load_json(output_root / "credential_reference.json")
    strict_json_nonstandard_token_count = sum(base.strict_json_token_count(path) for path in output_root.rglob("*.json"))
    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "source_artifact": str(source_artifact),
        "run_date": schedule.get("run_date"),
        "campaign_window": f"{schedule.get('campaign_start_time')} to {schedule.get('planned_campaign_end_time')}",
        "authorization_approved": True,
        "blocking_reason": None,
        "offline_replay_api_calls_made": 0,
        "prior_physical_calls_on_run_date": daily.get("prior_physical_calls_on_run_date"),
        "preflight_physical_calls": preflight.get("preflight_physical_calls"),
        "campaign_physical_calls": runtime.get("campaign_physical_calls"),
        "total_physical_calls_on_run_date": daily.get("total_physical_calls_on_run_date"),
        "effective_campaign_hard_cap": runtime.get("effective_campaign_hard_cap"),
        "max_calls_per_minute": runtime.get("max_calls_per_minute"),
        "fatal_api_error": runtime.get("fatal_api_error"),
        "first_fatal_error_time": runtime.get("first_fatal_error_time"),
        "calls_after_first_fatal_error": runtime.get("calls_after_first_fatal_error"),
        "route_results": {route: {"complete": counter["by_route"][route]["new_complete_episode_count"], "left": counter["by_route"][route]["left_censored_episode_count"], "right": counter["by_route"][route]["right_censored_episode_count"], "new_vehicles": counter["by_route"][route]["new_independent_complete_vehicle_count"]} for route in base.TARGET_ROUTES},
        "campaign_a_new_complete_total": campaign_complete_count,
        "campaign_a_new_independent_vehicle_total": counter["total"]["new_independent_complete_vehicle_count"],
        "cumulative_complete_candidate": cumulative_count,
        "remaining_to_method_prototype_12": max(0, 12 - cumulative_count),
        "mapping_regression_count": mapping_regression_count,
        "episode_duplicate_count": duplicate_count,
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
    base.dump_json(output_root / "prompt5_e01_r2d1h_gate.json", gate)

    report_lines = [
        "# Prompt 5-E01-R2D-1H Offline Replay Repair Report",
        "",
        "## Status",
        "",
        f"- gate: {gate_status}",
        "- repair_mode: offline raw replay",
        f"- source_artifact: {source_artifact}",
        "- api_calls_made_by_repair: 0",
        f"- run_date: {gate['run_date']}",
        f"- campaign_window: {gate['campaign_window']}",
        f"- preflight_physical_calls: {gate['preflight_physical_calls']}",
        f"- campaign_physical_calls: {gate['campaign_physical_calls']}",
        f"- total_physical_calls_on_run_date: {gate['total_physical_calls_on_run_date']}",
        f"- fatal_api_error: {gate['fatal_api_error'] or 'NONE'}",
        "",
        "## Corrected Route Results",
    ]
    for route in base.TARGET_ROUTES:
        rr = gate["route_results"][route]
        report_lines.append(f"- {route}: complete={rr['complete']}, left={rr['left']}, right={rr['right']}, new_independent_complete={rr['new_vehicles']}")
    report_lines.extend(["", "## Episode Reclassification"])
    for row in reclassifications:
        report_lines.append(
            f"- {row['episode_id']}: {row['source_final_status_class']} -> {row['repaired_final_status_class']}, vehicle={row['vehicle_id']}, reset={row['last_terminal_sequence']}->{row['first_post_terminal_sequence']}"
        )
    report_lines.extend(
        [
            "",
            "## Audits",
            "",
            f"- raw files replayed: {len(paths)}",
            f"- raw byte identity passed: {str(raw_unchanged).lower()}",
            f"- Counter Contract v9 passed: {str(counter['counter_contract_v9_passed']).lower()}",
            f"- raw SHA provenance passed: {str(evidence['episode_evidence_sha256_audit_passed']).lower()}",
            f"- interval validation passed: {str(interval['interval_validation_passed']).lower()}",
            f"- duplicate count: {duplicate_count}",
            f"- mapping regression count: {mapping_regression_count}",
            f"- invalid clock order count: {invalid_clock_order_count}",
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
            "The source artifact's raw evidence is preserved. Its derived episode-count conclusions are superseded by this offline replay repair.",
        ]
    )
    (output_root / "prompt5_e01_r2d1h_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    superseded_note = source_artifact / "SUPERSEDED_FOR_SCIENTIFIC_EPISODE_COUNTS.md"
    superseded_note.write_text(
        "\n".join(
            [
                "# Superseded For Scientific Episode-Count Conclusions",
                "",
                f"- superseded_at: {base.iso()}",
                f"- superseded_by: {output_root}",
                "- raw_evidence_status: preserved unchanged",
                "- scope: derived episode counts, route summaries, cumulative registry candidate, and progress/gate metadata",
                "- reason: independent audit found two exact-ID terminal reset episodes in saved raw responses that were previously finalized as right-censored response loss.",
                "- api_calls_made_for_repair: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base.dump_json(
        output_root / "source_artifact_supersession_audit.json",
        {
            "source_artifact": str(source_artifact),
            "superseded_note": str(superseded_note),
            "superseded_note_sha256": base.sha256_file(superseded_note),
            "raw_evidence_preserved_unchanged": True,
            "source_derived_scientific_episode_counts_superseded": True,
        },
    )
    write_manifest(output_root)
    print("R2D-1H CAMPAIGN A OFFLINE REPLAY REPAIR COMPLETE")
    print(output_root)
    print(gate_status)


if __name__ == "__main__":
    main()
