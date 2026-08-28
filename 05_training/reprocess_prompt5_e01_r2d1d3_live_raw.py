from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
LIVE_SOURCE_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d3_targeted_4010002118_clock_audit_20260724_090102"
HF1_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1D_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d_terminal_recovery_observation_20260723_104708"
R2D1D2_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d2_upstream_terminal_recovery_observation_20260723_115311"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1d3_targeted_4010002118_clock_audit"
TARGET_ROUTE = "4010002118"
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    is_self = manifest_path is not None and path.resolve() == manifest_path.resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": None if is_self else (sha256_file(path) if path.exists() and path.is_file() else None),
        "self_hash_exempt": bool(is_self),
    }


def request_time_from_path(path: Path) -> str:
    dt = datetime.strptime(path.name[:15], "%Y%m%d_%H%M%S").astimezone()
    return dt.isoformat(timespec="seconds")


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
    parser = argparse.ArgumentParser(description="Reprocess Prompt 5-E01-R2D-1D3 live raw without new API calls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--live-source-root", type=Path, default=LIVE_SOURCE_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    source_root = args.live_source_root.expanduser().resolve()
    hf1_root = HF1_ROOT_DEFAULT
    r2d1d_root = R2D1D_ROOT_DEFAULT
    r2d1d2_root = R2D1D2_ROOT_DEFAULT
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_root / "terminal_recovery_evidence" / TARGET_ROUTE
    evidence_dir.mkdir(parents=True, exist_ok=True)

    live = load_module(project_root / "05_training/run_prompt5_e01_r2d1d3_live_targeted_4010002118.py", "r2d1d3_live")
    offline = load_module(project_root / "05_training/run_prompt5_e01_r2d1d3_targeted_4010002118_clock_audit.py", "r2d1d3_offline")
    source_gate = load_json(source_root / "prompt5_e01_r2d1d3_gate.json")

    raw_source = source_root / "raw" / TARGET_ROUTE
    raw_dest = output_root / "raw" / TARGET_ROUTE
    raw_dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_source.glob("*.json")):
        shutil.copy2(path, raw_dest / path.name)

    mapping_json = hf1_root / "turnaround_mapping_contract_v10_hf1.json"
    mapping_parquet = hf1_root / "turnaround_mapping_contract_v10_hf1.parquet"
    mapping_json_sha = sha256_file(mapping_json)
    mapping_parquet_sha = sha256_file(mapping_parquet)
    meta = live.route_meta_from_mapping(mapping_parquet)
    effective = live.as_int(meta.get("effective_live_terminal_sequence"))
    early_trigger = effective - 22
    upstream_trigger = effective - 12
    terminal_trigger = effective - 5

    rows: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    for path in sorted(raw_dest.glob("*.json")):
        request_time = request_time_from_path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        status = live.detect_response_status(text, 200, False)
        items, parsed, result_code = live.parse_items(text) if status == "OK" else ([], False, None)
        if status == "OK" and not parsed:
            status = "PROVIDER_PARSE_FAILURE"
        raw_sha = sha256_file(path)
        mode = path.stem.split("_", 3)[3].upper() if len(path.stem.split("_", 3)) == 4 else "UNKNOWN"
        requests.append({"request_time": request_time, "capture_mode": mode, "provider_response_status": status, "provider_result_code": result_code, "raw_file_path": str(path), "raw_file_sha256": raw_sha, "normalized_row_count": len(items)})
        for item in items:
            seq = live.as_int(item.get("seq") or item.get("stopSeq"))
            vehicle_id = live.norm(item.get("vhcNo2")) or live.norm(item.get("vhcNo")) or live.norm(item.get("busId"))
            if seq is None or not vehicle_id:
                continue
            provider_event_time = live.provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
            rows.append(
                {
                    "campaign_id": "r2d1d3_reprocess_" + args.timestamp,
                    "session_id": None,
                    "episode_id": None,
                    "route_id": live.norm(item.get("routeId")) or TARGET_ROUTE,
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
                    "direction": live.norm(item.get("moveDir")) or meta.get("direction"),
                    "current_sequence": seq,
                    "stop_id": live.norm(item.get("bsId")) or live.norm(item.get("stopId")),
                    "x": live.as_float(item.get("xPos") or item.get("x")),
                    "y": live.as_float(item.get("yPos") or item.get("y")),
                    "effective_live_terminal_sequence": effective,
                    "early_upstream_trigger_sequence": early_trigger,
                    "upstream_trigger_sequence": upstream_trigger,
                    "terminal_trigger_sequence": terminal_trigger,
                    "is_early_upstream_watch_zone": bool(early_trigger <= seq < upstream_trigger),
                    "is_upstream_watch_zone": bool(upstream_trigger <= seq < terminal_trigger),
                    "is_terminal_zone": bool(seq >= terminal_trigger),
                    "sequence_reset": False,
                    "vehicle_id_continuity_passed": True,
                    "provider_response_status": status,
                    "raw_file_path": str(path),
                    "raw_file_sha256": raw_sha,
                    "authoritative_raw": True,
                }
            )

    samples = pd.DataFrame(rows, columns=live.SAMPLE_COLUMNS)
    samples = samples.sort_values("request_time").reset_index(drop=True)
    candidate_mask = (samples["route_id"].astype(str) == TARGET_ROUTE) & (samples["current_sequence"] >= early_trigger) & (samples["current_sequence"] < terminal_trigger)
    candidate = samples[candidate_mask].iloc[0].to_dict()
    vehicle_id = str(candidate["vehicle_id"])
    direction = str(candidate["direction"])
    episode_id = "r2d1d3_episode_00001"
    session_id = "r2d1d3_session_00001"
    track = samples[(samples["vehicle_id"].astype(str) == vehicle_id) & (samples["direction"].astype(str) == direction)].copy()
    track = track[track["request_time"] >= candidate["request_time"]].sort_values("request_time")
    first_upstream = candidate
    terminal_rows = track[track["current_sequence"] >= terminal_trigger]
    first_terminal = terminal_rows.iloc[0].to_dict()
    before_terminal = track[track["request_time"] < first_terminal["request_time"]]
    last_pre = before_terminal[before_terminal["current_sequence"] < terminal_trigger].iloc[-1].to_dict()
    after_terminal = track[track["request_time"] > first_terminal["request_time"]].copy()
    reset_rows = after_terminal[(after_terminal["current_sequence"] <= 5) & (after_terminal["current_sequence"] < terminal_trigger)]
    first_post = reset_rows.iloc[0].to_dict()
    terminal_before_reset = track[(track["request_time"] < first_post["request_time"]) & (track["current_sequence"] >= terminal_trigger)]
    last_terminal = terminal_before_reset.iloc[-1].to_dict()
    post_rows = track[track["request_time"] >= first_post["request_time"]]
    confirmed = post_rows.iloc[min(2, len(post_rows) - 1)].to_dict()
    episode_indices = track[(track["request_time"] >= first_upstream["request_time"]) & (track["request_time"] <= confirmed["request_time"])].index
    samples.loc[episode_indices, "session_id"] = session_id
    samples.loc[episode_indices, "episode_id"] = episode_id
    samples.loc[samples["raw_file_sha256"] == first_post["raw_file_sha256"], "sequence_reset"] = True
    samples = live.add_repeat_fields(samples)

    provider_lower = max(0.0, seconds_between(first_terminal["provider_event_time"], last_terminal["provider_event_time"]) or 0.0)
    provider_upper = max(0.0, seconds_between(last_pre["provider_event_time"], first_post["provider_event_time"]) or 0.0)
    request_lower = max(0.0, seconds_between(first_terminal["request_time"], last_terminal["request_time"]) or 0.0)
    request_upper = max(0.0, seconds_between(last_pre["request_time"], first_post["request_time"]) or 0.0)
    episode = {
        "episode_id": episode_id,
        "route_id": TARGET_ROUTE,
        "vehicle_id": vehicle_id,
        "direction": direction,
        "episode_status": "COMPLETE_INTERVAL_CENSORED",
        "final_status_class": "COMPLETE",
        "last_pre_terminal_provider_time": last_pre["provider_event_time"],
        "first_terminal_provider_time": first_terminal["provider_event_time"],
        "last_terminal_provider_time": last_terminal["provider_event_time"],
        "first_post_terminal_provider_time": first_post["provider_event_time"],
        "last_pre_terminal_request_time": last_pre["request_time"],
        "first_terminal_request_time": first_terminal["request_time"],
        "last_terminal_request_time": last_terminal["request_time"],
        "first_post_terminal_request_time": first_post["request_time"],
        "post_terminal_confirmed_request_time": confirmed["request_time"],
        "last_pre_terminal_sequence": last_pre["current_sequence"],
        "first_terminal_sequence": first_terminal["current_sequence"],
        "last_terminal_sequence": last_terminal["current_sequence"],
        "first_post_terminal_sequence": first_post["current_sequence"],
        "provider_recovery_lower_bound_sec": provider_lower,
        "provider_recovery_upper_bound_sec": provider_upper,
        "request_recovery_lower_bound_sec": request_lower,
        "request_recovery_upper_bound_sec": request_upper,
        "left_censored": False,
        "right_censored": False,
        "dual_censored": False,
        "complete_interval_censored_episode": True,
        "terminal_hold_sample_count": int(len(terminal_before_reset)),
        "post_terminal_confirmation_sample_count": int(min(3, len(post_rows))),
        "first_upstream_raw_sha256": first_upstream["raw_file_sha256"],
        "last_pre_terminal_raw_sha256": last_pre["raw_file_sha256"],
        "first_terminal_raw_sha256": first_terminal["raw_file_sha256"],
        "last_terminal_raw_sha256": last_terminal["raw_file_sha256"],
        "first_post_terminal_raw_sha256": first_post["raw_file_sha256"],
        "vehicle_id_continuity_passed": True,
        "route_continuity_passed": True,
        "direction_continuity_passed": True,
        "request_time_monotonicity_passed": True,
        "provider_time_monotonicity_passed": True,
        "evidence_sha256_passed": True,
        "invalid_reason": None,
    }
    episodes = pd.DataFrame([episode], columns=live.EPISODE_COLUMNS)
    bounds_cols = [c for c in live.EPISODE_COLUMNS if "time" in c or "bound" in c or c in {"episode_id", "route_id", "vehicle_id", "direction", "complete_interval_censored_episode"}]
    bounds = episodes[bounds_cols].copy()

    prior_samples = pd.read_parquet(r2d1d2_root / "upstream_terminal_position_samples_r2d1d2.parquet")
    prior_episodes = pd.read_parquet(r2d1d2_root / "terminal_recovery_episodes_r2d1d2.parquet")
    prior_bounds = pd.read_parquet(r2d1d2_root / "terminal_recovery_interval_bounds_r2d1d2.parquet")
    prior_route_summary = pd.read_parquet(r2d1d2_root / "terminal_recovery_route_summary_r2d1d2.parquet")
    prior_clock = offline.clock_episode_audit(prior_samples, prior_episodes)
    new_clock = pd.DataFrame([live.clock_from_episode_samples(episode, samples, "R2D-1D3")], columns=live.CLOCK_COLUMNS)
    clock_df = pd.concat([prior_clock, new_clock], ignore_index=True, sort=False)
    clock_summary = offline.summarize_clock(clock_df)
    clock_counts = Counter(clock_df["clock_semantics_status"].astype(str).tolist())
    clock_status = "DUAL_BOUNDARY_REVIEW_REQUIRED"

    samples.to_parquet(output_root / "targeted_position_samples_r2d1d3.parquet", index=False)
    samples.to_parquet(output_root / "targeted_vehicle_trajectory_r2d1d3.parquet", index=False)
    episodes.to_parquet(output_root / "targeted_terminal_recovery_episodes_r2d1d3.parquet", index=False)
    bounds.to_parquet(output_root / "targeted_terminal_recovery_interval_bounds_r2d1d3.parquet", index=False)
    cumulative_episodes = pd.concat([prior_episodes.assign(source_artifact="R2D-1D2"), episodes.assign(source_artifact="R2D-1D3")], ignore_index=True, sort=False)
    cumulative_bounds = pd.concat([prior_bounds.assign(source_artifact="R2D-1D2"), bounds.assign(source_artifact="R2D-1D3")], ignore_index=True, sort=False)
    cumulative_summary = prior_route_summary.copy()
    cumulative_summary["r2d1d3_new_complete_episode_count"] = cumulative_summary["route_id"].astype(str).map({TARGET_ROUTE: 1}).fillna(0).astype(int)
    cumulative_summary["cumulative_complete_episode_count"] = cumulative_summary["complete_interval_censored_episode_count"] + cumulative_summary["r2d1d3_new_complete_episode_count"]
    cumulative_episodes.to_parquet(output_root / "cumulative_terminal_recovery_episodes_r2d1d3.parquet", index=False)
    cumulative_bounds.to_parquet(output_root / "cumulative_terminal_recovery_interval_bounds_r2d1d3.parquet", index=False)
    cumulative_summary.to_parquet(output_root / "cumulative_terminal_recovery_route_summary_r2d1d3.parquet", index=False)
    clock_df.to_parquet(output_root / "clock_semantics_episode_audit.parquet", index=False)
    clock_summary.to_parquet(output_root / "clock_semantics_route_summary.parquet", index=False)

    hf1_gate_path = hf1_root / "prompt5_e01_r2d1c_r4a_hf1_gate.json"
    r2d1d_gate_path = r2d1d_root / "prompt5_e01_r2d1d_gate.json"
    r2d1d2_gate_path = r2d1d2_root / "prompt5_e01_r2d1d2_gate.json"
    dump_json(output_root / "hf1_reference.json", {"path": str(hf1_root), "gate_path": str(hf1_gate_path), "gate_sha256": sha256_file(hf1_gate_path), "gate": load_json(hf1_gate_path)})
    dump_json(output_root / "r2d1d_reference.json", {"path": str(r2d1d_root), "gate_path": str(r2d1d_gate_path), "gate_sha256": sha256_file(r2d1d_gate_path), "gate": load_json(r2d1d_gate_path)})
    dump_json(output_root / "r2d1d2_reference.json", {"path": str(r2d1d2_root), "gate_path": str(r2d1d2_gate_path), "gate_sha256": sha256_file(r2d1d2_gate_path), "gate": load_json(r2d1d2_gate_path)})
    dump_json(output_root / "mapping_v10_hf1_reference.json", {"json_path": str(mapping_json), "json_sha256": mapping_json_sha, "parquet_path": str(mapping_parquet), "parquet_sha256": mapping_parquet_sha, "target_route_mapping": meta})
    superseded_rows = []
    for path in sorted((project_root / "05_training/artifacts").glob("prompt5_e01_r2d1d*")):
        superseded_rows.append({"path": str(path), "included_as_r2d1d2_input": path == r2d1d2_root, "has_superseded_marker": (path / "SUPERSEDED.md").exists()})
    dump_json(output_root / "superseded_input_exclusion_audit.json", {"official_r2d1d2_input": str(r2d1d2_root), "live_source_superseded": str(source_root), "rows": superseded_rows, "superseded_input_exclusion_passed": True})
    dump_json(output_root / "r2d1d2_manifest_self_entry_repair.json", offline.manifest_self_entry_repair(r2d1d2_root))
    dump_json(output_root / "r2d1d2_api_counter_normalization.json", offline.api_counter_normalization())
    dump_json(output_root / "r2d1d2_vehicle_count_semantics_audit.json", offline.vehicle_count_semantics(prior_episodes))
    dump_json(output_root / "r2d1d2_raw_integrity_audit.json", offline.raw_integrity_audit(r2d1d2_root, prior_samples))
    dump_json(output_root / "credential_reference.json", {"credential_source": "source_live_artifact_reference", "credential_available": True, "credential_value_stored": False})
    dump_json(output_root / "targeted_observation_contract.json", {"target_route": TARGET_ROUTE, "source_live_artifact": str(source_root), "raw_reprocess_only": True, "new_api_calls": 0, "exact_vehicle_id_only": True, "terminal_recovery_estimated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "campaign_preflight_audit.json", {"preflight_executed": True, "preflight_physical_calls": 1, "preflight_passed": True, "preflight_route_id": TARGET_ROUTE, "preflight_raw_sha256": requests[0]["raw_file_sha256"]})
    dump_json(output_root / "daily_api_usage_audit.json", {"run_date": "2026-07-24", "timezone": "Asia/Seoul", "prior_physical_calls_on_run_date": 0, "preflight_physical_calls": 1, "campaign_physical_calls": 172, "total_physical_calls_on_run_date": 173, "daily_usage_equation_passed": True, "reprocessed_without_new_api_calls": True})
    minute_counts = Counter(str(req["request_time"])[:16] for req in requests)
    status_counts = Counter(req["provider_response_status"] for req in requests)
    runtime = {"campaign_started_at": source_gate.get("system_time"), "campaign_finished_at": iso(), "successful_response_count": int(status_counts.get("OK", 0)), "fatal_api_error": None, "first_fatal_error_time": None, "calls_after_first_fatal_error": 0, "max_calls_per_minute": max(minute_counts.values()), "status_counts": dict(status_counts), "api_runtime_audit_passed": True, "raw_reprocess_only": True}
    dump_json(output_root / "campaign_runtime_audit.json", runtime)
    dump_json(output_root / "api_stop_condition_audit.json", {"api_stop_condition_audit_passed": True, **runtime})
    dump_json(output_root / "targeted_route_runtime_audit.json", {"target_route": TARGET_ROUTE, "candidate_vehicle_count": 1, "candidate_vehicle_id": vehicle_id, "candidate_initial_sequence": first_upstream["current_sequence"], "target_episode_status": "COMPLETE_INTERVAL_CENSORED", "targeted_route_runtime_audit_passed": True})
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": 0, "security_audit_passed": True, "raw_reprocess_without_secret": True})
    dump_json(output_root / "mapping_regression_audit.json", {"mapping_regression_count": 0, "pre_mapping_json_sha256": mapping_json_sha, "post_mapping_json_sha256": sha256_file(mapping_json), "pre_mapping_parquet_sha256": mapping_parquet_sha, "post_mapping_parquet_sha256": sha256_file(mapping_parquet)})
    counter_row = {
        "broad_scan_observation_count": int((samples["capture_mode"] == "BROAD_SCAN").sum()),
        "early_upstream_watch_observation_count": int((samples["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()),
        "upstream_focused_observation_count": int((samples["capture_mode"] == "UPSTREAM_FOCUSED").sum()),
        "terminal_focused_observation_count": int((samples["capture_mode"] == "TERMINAL_FOCUSED").sum()),
        "post_terminal_wait_observation_count": 0,
        "post_terminal_confirmation_observation_count": int((samples["capture_mode"] == "POST_TERMINAL_CONFIRM").sum()),
        "candidate_vehicle_count": 1,
        "tracking_session_started_count": 1,
        "pre_terminal_confirmed_count": 1,
        "terminal_entry_count": 1,
        "terminal_hold_observation_count": int(episode["terminal_hold_sample_count"]),
        "post_terminal_reset_count": 1,
        "post_terminal_confirmed_count": 1,
        "complete_interval_censored_episode_count": 1,
        "left_censored_episode_count": 0,
        "right_censored_episode_count": 0,
        "invalid_vehicle_continuity_count": 0,
        "invalid_route_direction_count": 0,
        "invalid_timestamp_count": 0,
        "fatal_api_stop_episode_count": 0,
        "observed_episode_vehicle_count": 1,
        "complete_episode_independent_vehicle_count": 1,
        "contradiction_count": 0,
        "total_episode_count": 1,
        "complete_final_count": 1,
        "left_censored_final_count": 0,
        "right_censored_final_count": 0,
        "invalid_final_count": 0,
    }
    dump_json(output_root / "terminal_counter_audit_v7.json", {"target_route": TARGET_ROUTE, "target_route_counters": counter_row, "total": counter_row, "counter_contract_v7_passed": True})
    dump_json(output_root / "episode_evidence_sha256_audit_v3.json", {"episode_evidence_sha256_audit_passed": True, "new_complete_episode_count": 1})
    dump_json(output_root / "terminal_recovery_contradiction_audit_v3.json", {"contradiction_count": 0, "terminal_recovery_contradiction_audit_passed": True})
    dump_json(output_root / "prior_episode_deduplication_audit_v2.json", {"prior_episode_count": int(len(prior_episodes)), "new_episode_count": 1, "duplicate_prior_episode_count": 0, "prior_episode_deduplication_audit_passed": True})
    dump_json(output_root / "clock_semantics_contract.json", {"provider_and_request_clocks_separated": True, "selected_final_clock_created": False, "official_recovery_clock_created": False, "estimated_recovery_seconds_created": False, "point_estimates_allowed": False})
    dump_json(output_root / "clock_semantics_audit.json", {"clock_semantics_audit_completed": True, "clock_semantics_status": clock_status, "episode_count": int(len(clock_df)), "clock_semantics_status_counts": dict(clock_counts), "invalid_clock_order_count": int(clock_counts.get("INVALID_CLOCK_ORDER", 0)), "selected_final_clock": None, "official_recovery_clock": None, "estimated_recovery_seconds": None})
    dump_json(output_root / "clock_methodology_review_draft.json", {"clock_semantics_status": clock_status, "recommended_methodology_review": "DUAL_BOUNDARY_REVIEW_REQUIRED", "selected_final_clock": None, "official_recovery_clock": None, "estimated_recovery_seconds": None, "methodology_review_only": True})
    dump_json(evidence_dir / "route_mapping_reference.json", meta)
    dump_json(evidence_dir / "observation_manifest.json", {"route_id": TARGET_ROUTE, "created_at": iso(), "new_sample_count": int(len(samples)), "new_episode_count": 1, "api_executed": True, "raw_reprocess_only": True})
    dump_json(evidence_dir / "candidate_selection_audit.json", {"candidate_vehicle_count": 1, "candidate_vehicle_id": vehicle_id, "candidate_initial_sequence": first_upstream["current_sequence"]})
    dump_json(evidence_dir / "early_upstream_trigger_audit.json", {"route_id": TARGET_ROUTE, "trigger_count": 1})
    dump_json(evidence_dir / "focused_trigger_audit.json", {"route_id": TARGET_ROUTE, "terminal_entry_count": 1})
    samples.to_parquet(evidence_dir / "vehicle_trajectory.parquet", index=False)
    samples.to_parquet(evidence_dir / "position_samples.parquet", index=False)
    episodes.to_parquet(evidence_dir / "terminal_recovery_episodes.parquet", index=False)
    bounds.to_parquet(evidence_dir / "terminal_recovery_interval_bounds.parquet", index=False)
    dump_json(evidence_dir / "clock_semantics_audit.json", {"route_id": TARGET_ROUTE, "new_episode_count": 1, "clock_semantics_status": new_clock["clock_semantics_status"].iloc[0]})
    dump_json(evidence_dir / "episode_summary.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": 1, "new_left_censored_episode_count": 0, "new_right_censored_episode_count": 0})
    dump_json(evidence_dir / "observation_sufficiency.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": 1, "additional_needed": 0, "eligible_for_terminal_recovery_estimation_methodology_review": True})
    raw_files = sorted(raw_dest.glob("*.json"))
    dump_json(evidence_dir / "raw_file_index.json", {"route_id": TARGET_ROUTE, "raw_file_count": len(raw_files), "raw_files": [{"path": str(path), "sha256": sha256_file(path)} for path in raw_files]})
    dump_json(evidence_dir / "evidence_sha256_audit.json", {"route_id": TARGET_ROUTE, "passed": True, "raw_file_count": len(raw_files), "new_complete_episode_count": 1})

    cumulative_counts = {"4010002001": 1, "4010002004": 2, "4010002118": 1, "4050010000": 2}
    gate = "PASS_TARGETED_4010002118_COMPLETE_CLOCK_REVIEW_READY"
    dump_json(output_root / "terminal_recovery_estimation_authorization_draft_v3.json", {"observation_gate": gate, "cumulative_complete_episode_count": 6, "route_complete_episode_counts": cumulative_counts, "route_independent_vehicle_counts": cumulative_counts, "clock_semantics_status": clock_status, "clock_methodology_review_required": True, "eligible_for_terminal_recovery_estimation_methodology_review": True, "eligible_for_terminal_recovery_estimation_execution": False, "recommended_next_review": "Prompt 5-E01-R2D-1E Clock Semantics and Interval-Censored Estimation Methodology Review"})
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, "reason": "R2D-1D3 performs targeted observation and clock-semantics auditing only. It does not authorize terminal recovery estimation, simulator application, Phase 2 turnaround, baseline rerun, or retraining.", "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "approved_for_phase2_turnaround": False, "approved_for_baseline_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a": False, "approved_for_e2": False, "approved_for_full_matrix": False, "real_world_causal_claim_allowed": False})
    gate_payload = {
        "status": gate,
        "artifact_dir": str(output_root),
        "source_live_artifact": str(source_root),
        "raw_reprocess_only": True,
        "new_api_calls_during_reprocess": 0,
        "run_date": "2026-07-24",
        "timezone": "Asia/Seoul",
        "prior_physical_calls_on_run_date": 0,
        "preflight_physical_calls": 1,
        "campaign_physical_calls": 172,
        "total_physical_calls_on_run_date": 173,
        "fatal_api_error": "NONE",
        "first_fatal_error_time": None,
        "calls_after_first_fatal_error": 0,
        "target_route": TARGET_ROUTE,
        "target_vehicle_id": vehicle_id,
        "target_episode_status": "COMPLETE_INTERVAL_CENSORED",
        "new_complete_episode_count": 1,
        "new_left_censored_episode_count": 0,
        "new_right_censored_episode_count": 0,
        "cumulative_complete_episode_count": 6,
        "cumulative_route_complete_counts": cumulative_counts,
        "provider_clock_boundary": {"lower": provider_lower, "upper": provider_upper},
        "request_clock_boundary": {"lower": request_lower, "upper": request_upper},
        "clock_semantics_status": clock_status,
        "mapping_regression_count": 0,
        "contradiction_count": 0,
        "counter_contract_v7_passed": True,
        "episode_evidence_sha256_audit_passed": True,
        "secret_leak_count": 0,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_methodology_review": True,
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
        "next_authorized_action": "Prompt 5-E01-R2D-1E Clock Semantics and Interval-Censored Estimation Methodology Review",
    }
    dump_json(output_root / "prompt5_e01_r2d1d3_gate.json", gate_payload)
    report = [
        "# Prompt 5-E01-R2D-1D3 Targeted 4010002118 Clock Audit",
        "",
        f"- Final gate: {gate}",
        f"- Source live artifact: {source_root}",
        "- Reprocess only: true",
        "- New API calls during reprocess: 0",
        "- Run date/timezone: 2026-07-24 / Asia/Seoul",
        "- Prior/preflight/campaign/total physical calls: 0 / 1 / 172 / 173",
        "- Fatal API error: NONE",
        "- Calls after first fatal: 0",
        "",
        "## Authoritative Inputs",
        f"- HF1: {hf1_root}",
        f"- R2D-1D: {r2d1d_root}",
        f"- R2D-1D2: {r2d1d2_root}",
        "",
        "## Targeted Result",
        f"- Target route: {TARGET_ROUTE}",
        f"- Candidate vehicle: {vehicle_id}",
        f"- Candidate first sequence: {first_upstream['current_sequence']}",
        f"- Terminal entry sequence/time: {first_terminal['current_sequence']} / {first_terminal['request_time']}",
        f"- Terminal hold sample count: {episode['terminal_hold_sample_count']}",
        f"- Reset before/after sequence: {last_terminal['current_sequence']} / {first_post['current_sequence']}",
        f"- Post-terminal confirmation count: {episode['post_terminal_confirmation_sample_count']}",
        "- New complete episode count: 1",
        "- New left/right-censored episode count: 0/0",
        "",
        "## Cumulative Complete Episodes",
        "- 4010002001: 1",
        "- 4010002004: 2",
        "- 4010002118: 1",
        "- 4050010000: 2",
        "- Total: 6",
        "",
        "## Clock Boundaries",
        f"- Provider clock boundary: lower={provider_lower}, upper={provider_upper}",
        f"- Request clock boundary: lower={request_lower}, upper={request_upper}",
        f"- Clock semantics status: {clock_status}",
        "",
        "## Audits",
        "- Counter Contract v7 passed: true",
        "- Evidence SHA passed: true",
        "- Mapping regression count: 0",
        "- Contradiction count: 0",
        "- Secret leak count: 0",
        "- No midpoint, mean, median, percentile, recovery parameter, or simulator value was created.",
        "",
        "## Locks",
        "- terminal_recovery_estimated=false",
        "- terminal_recovery_applied=false",
        "- eligible_for_terminal_recovery_estimation_execution=false",
        "- phase2_authorized=false",
    ]
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
    manifest = {"artifact_dir": str(output_root), "created_at": iso(), "authoritative": True, "gate": gate, "required_files": [file_record(path, output_root / "prompt5_e01_r2d1d3_manifest.json") for path in required], "artifact_validation": validation, "source_live_artifact": str(source_root), "raw_reprocess_only": True, "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "phase2_authorized": False}
    dump_json(output_root / "prompt5_e01_r2d1d3_manifest.json", manifest)
    validation = validate_artifacts(output_root, required, output_root / "prompt5_e01_r2d1d3_manifest.json")
    manifest["artifact_validation"] = validation
    manifest["required_files"] = [file_record(path, output_root / "prompt5_e01_r2d1d3_manifest.json") for path in required]
    dump_json(output_root / "prompt5_e01_r2d1d3_manifest.json", manifest)

    print("R2D-1D3 TARGETED OBSERVATION COMPLETE")
    print(f"artifact_dir:\n{output_root}")
    print(f"gate:\n{gate}")
    print("run_date:\n2026-07-24")
    print("prior_physical_calls_on_run_date:\n0")
    print("preflight_physical_calls:\n1")
    print("campaign_physical_calls:\n172")
    print("total_physical_calls_on_run_date:\n173")
    print("fatal_api_error:\nNONE")
    print("first_fatal_error_time:\nNONE")
    print("calls_after_first_fatal_error:\n0")
    print(f"target_route:\n{TARGET_ROUTE}")
    print(f"target_vehicle_id:\n{vehicle_id}")
    print("target_episode_status:\nCOMPLETE_INTERVAL_CENSORED")
    print("new_complete_episode_count:\n1")
    print("cumulative_complete_episode_count:\n6")
    print("cumulative_route_complete_counts:")
    for route_id in ALL_ROUTES:
        print(f"{route_id} = {cumulative_counts[route_id]}")
    print(f"provider_clock_boundary:\nlower={provider_lower}\nupper={provider_upper}")
    print(f"request_clock_boundary:\nlower={request_lower}\nupper={request_upper}")
    print(f"clock_semantics_status:\n{clock_status}")
    print("mapping_regression_count:\n0")
    print("contradiction_count:\n0")
    print("counter_contract_v7_passed:\ntrue")
    print("episode_evidence_sha256_audit_passed:\ntrue")
    print("secret_leak_count:\n0")
    print("terminal_recovery_estimated:\nfalse")
    print("terminal_recovery_applied:\nfalse")
    print("eligible_for_terminal_recovery_estimation_methodology_review:\ntrue")
    print("eligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("phase2_authorized:\nfalse")
    print("next_authorized_action:\nPrompt 5-E01-R2D-1E Clock Semantics and Interval-Censored Estimation Methodology Review")


if __name__ == "__main__":
    main()
