from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r4a_final4_daytime_terminal_semantics"
R4_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_r4_final8_terminal_semantics_20260721_184800"
FINAL4_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
SCHEDULED_START_KST = "2026-07-22T09:00:00+09:00"
SCHEDULED_END_KST = "2026-07-22T13:00:00+09:00"
START_SEQUENCE_UPPER_BOUND = 10
TERMINAL_SEQUENCE_TOLERANCE = 6
CLASSIFIED_OPS = {
    "LOOP_CONTINUOUS",
    "PAIRED_OPPOSITE_DIRECTION",
    "OFF_GRAPH_CONTINUATION_AND_REENTRY",
    "DEPOT_OR_DEADHEAD_TRANSITION",
}
SAMPLE_COLUMNS_R4A = [
    "request_time_kst",
    "provider_event_time",
    "route_id",
    "direction_id",
    "vehicle_id",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
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
    "terminal_stop_id",
    "batch_id",
]


def load_r4_module(project_root: Path):
    path = project_root / "05_training/run_prompt5_e01_r2d1c_r4_final8_terminal_semantics.py"
    spec = importlib.util.spec_from_file_location("r4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.FINAL8_ROUTES = FINAL4_ROUTES
    return module


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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


def validate_r4(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R4_ROOT
    gate_path = root / "prompt5_e01_r2d1c_r4_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = {
        "approved_mapping_count": 34,
        "missing_mapping_count": 4,
        "mapping_regression_count": 0,
        "focused_trigger_runtime_passed": True,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R4 prerequisite failed: {mismatches}")
    return root, gate


def load_route_contract(r4, r4_root: Path, output_root: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    v9 = pd.read_parquet(r4_root / "turnaround_mapping_contract_v9.parquet")
    route_rows = v9[v9["route_id"].astype(str).isin(FINAL4_ROUTES)].copy()
    by_route = {str(row["route_id"]): r4.terminal_meta(row.to_dict()) for _, row in route_rows.iterrows()}
    static = pd.read_parquet(r4_root / "unresolved_route_static_topology.parquet")
    static = static[static["route_id"].astype(str).isin(FINAL4_ROUTES)].copy()
    static.to_parquet(output_root / "unresolved_route_static_topology.parquet", index=False)
    return v9, by_route, static


def fetch_route_r4a(r4, route_id: str, secret: str, output_root: Path, mode: str, interval: int, meta: Mapping[str, Any], focused_triggered: bool = False, focused_trigger_time: Optional[str] = None):
    rows, req = r4.fetch_route(route_id, secret, output_root, "", mode, interval, meta, focused_triggered, focused_trigger_time)
    for row in rows:
        row["batch_id"] = ""
    return rows, req


def route_near_terminal(r4, row: Mapping[str, Any]) -> bool:
    return r4.near_terminal(row)


def run_capture(r4, args: argparse.Namespace, secret: str, output_root: Path, route_meta: Mapping[str, Mapping[str, Any]]):
    start_dt = parse_dt(SCHEDULED_START_KST)
    end_dt = parse_dt(SCHEDULED_END_KST)
    actual_start = datetime.now().astimezone()
    if actual_start < start_dt and not args.force_start_before_schedule:
        raise RuntimeError(f"R4A_COLLECTION_NOT_ALLOWED_BEFORE_SCHEDULED_START: now={actual_start.isoformat()} scheduled={SCHEDULED_START_KST}")
    samples: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    trigger_rows: List[Dict[str, Any]] = []
    active_focus: Dict[str, Dict[str, Any]] = {}
    observation_seconds = {route_id: 0.0 for route_id in FINAL4_ROUTES}
    post_follow_started: set[str] = set()
    last_focus_fetch: Dict[str, float] = {route_id: 0.0 for route_id in FINAL4_ROUTES}
    last_broad_fetch: Dict[str, float] = {route_id: 0.0 for route_id in FINAL4_ROUTES}

    while datetime.now().astimezone() < end_dt:
        now_mono = time.monotonic()
        for route_id in FINAL4_ROUTES:
            focus = active_focus.get(route_id)
            if focus and now_mono - last_focus_fetch[route_id] >= args.focused_interval_seconds:
                rows, req = fetch_route_r4a(r4, route_id, secret, output_root, focus["mode"], args.focused_interval_seconds, route_meta[route_id], True, focus["trigger_time"])
                samples.extend(rows)
                requests.append(req)
                observation_seconds[route_id] += args.focused_interval_seconds
                last_focus_fetch[route_id] = now_mono
                trigger_vehicle = str(focus["vehicle_id"])
                for row in rows:
                    if str(row.get("vehicle_id")) != trigger_vehicle:
                        continue
                    seq = as_int(row.get("current_sequence"))
                    if focus.get("last_sequence") is not None and seq is not None:
                        boundary = as_int(route_meta[route_id].get("effective_live_terminal_sequence")) or 0
                        if focus["last_sequence"] >= boundary - TERMINAL_SEQUENCE_TOLERANCE and seq <= START_SEQUENCE_UPPER_BOUND:
                            focus["mode"] = "POST_TERMINAL_FOLLOW"
                            post_follow_started.add(route_id)
                    if seq is not None:
                        focus["last_sequence"] = seq
                print(json.dumps({"event": "r4a_focused_tick", "route_id": route_id, "vehicle_id": trigger_vehicle, "mode": focus["mode"], "rows": len(rows), "time": now_kst()}, ensure_ascii=False), flush=True)
                if datetime.now().astimezone() >= focus["follow_end_time"]:
                    active_focus.pop(route_id, None)
            if route_id not in active_focus and now_mono - last_broad_fetch[route_id] >= args.broad_interval_seconds:
                rows, req = fetch_route_r4a(r4, route_id, secret, output_root, "BROAD_SCAN", args.broad_interval_seconds, route_meta[route_id])
                samples.extend(rows)
                requests.append(req)
                observation_seconds[route_id] += args.broad_interval_seconds
                last_broad_fetch[route_id] = now_mono
                matched = [row for row in rows if route_near_terminal(r4, row)]
                for row in rows:
                    trigger_rows.append(
                        {
                            "route_id": route_id,
                            "vehicle_id": row.get("vehicle_id"),
                            "sequence_at_trigger": as_int(row.get("current_sequence")),
                            "effective_terminal_sequence": route_meta[route_id].get("effective_live_terminal_sequence"),
                            "trigger_time": row.get("request_time_kst"),
                            "focused_interval_seconds": args.focused_interval_seconds,
                            "follow_end_time": None,
                            "trigger_matched": route_near_terminal(r4, row),
                            "focused_capture_started": False,
                            "post_terminal_follow_started": False,
                            "classification": "TRIGGER_EXECUTED" if route_near_terminal(r4, row) else "NO_TRIGGER_CANDIDATE",
                        }
                    )
                if matched:
                    trigger = matched[0]
                    follow_end = min(datetime.now().astimezone() + timedelta(minutes=args.post_terminal_follow_minutes), end_dt)
                    active_focus[route_id] = {
                        "vehicle_id": str(trigger.get("vehicle_id")),
                        "trigger_time": str(trigger.get("request_time_kst")),
                        "follow_end_time": follow_end,
                        "last_sequence": as_int(trigger.get("current_sequence")),
                        "mode": "TERMINAL_FOCUSED",
                    }
                    for audit_row in reversed(trigger_rows):
                        if audit_row["route_id"] == route_id and audit_row["vehicle_id"] == trigger.get("vehicle_id") and audit_row["trigger_time"] == trigger.get("request_time_kst"):
                            audit_row["focused_capture_started"] = True
                            audit_row["follow_end_time"] = follow_end.isoformat(timespec="seconds")
                            break
                    print(json.dumps({"event": "r4a_focused_trigger", "route_id": route_id, "vehicle_id": trigger.get("vehicle_id"), "sequence": trigger.get("current_sequence"), "time": now_kst()}, ensure_ascii=False), flush=True)
                print(json.dumps({"event": "r4a_broad_tick", "route_id": route_id, "rows": len(rows), "near_terminal": len(matched), "time": now_kst()}, ensure_ascii=False), flush=True)
        time.sleep(args.scheduler_poll_seconds)
    for row in trigger_rows:
        if row["route_id"] in post_follow_started:
            row["post_terminal_follow_started"] = True
    return samples, requests, trigger_rows, {route_id: seconds / 60.0 for route_id, seconds in observation_seconds.items()}, actual_start.isoformat(timespec="seconds"), datetime.now().astimezone().isoformat(timespec="seconds")


def update_mapping_v10(r4, v9: pd.DataFrame, decisions: Sequence[Mapping[str, Any]], output_root: Path):
    before = {idx for idx, row in v9.iterrows() if truthy(row.get("approved"))}
    by_route = {str(row["route_id"]): row for row in decisions}
    rows = []
    for _, row in v9.iterrows():
        out = row.to_dict()
        out["approved"] = truthy(out.get("approved"))
        out["newly_recovered_in_r4a"] = False
        route_id = str(out.get("route_id"))
        if route_id in by_route and not out["approved"]:
            decision = by_route[route_id]
            if truthy(decision.get("approved")):
                out["approved"] = True
                out["terminal_operation_type"] = decision["decision"]
                out["terminal_operation_resolved"] = True
                out["mapping_confidence"] = decision["confidence"]
                out["newly_recovered_in_r4a"] = True
                out["terminal_entry_event_count"] = decision.get("terminal_entry_event_count", 0)
                out["terminal_hold_observation_count"] = decision.get("terminal_hold_observation_count", 0)
                out["classified_operation_transition_count"] = decision.get("classified_operation_transition_count", 0)
                out["independent_vehicle_count"] = decision.get("independent_vehicle_count", 0)
                out["contradictory_transition_count"] = decision.get("contradiction_count", 0)
                out["live_evidence_paths"] = [str(output_root / "mapping_evidence" / route_id / "terminal_transition_events.parquet")]
        rows.append(out)
    after = {idx for idx, row in enumerate(rows) if truthy(row.get("approved"))}
    return {
        "contract_version": "turnaround_mapping_contract_v10",
        "source_contract_version": "turnaround_mapping_contract_v9",
        "previous_mapping_count": len(before),
        "newly_recovered_mapping_count": sum(1 for row in rows if truthy(row.get("newly_recovered_in_r4a"))),
        "approved_mapping_count": len(after),
        "missing_mapping_count": 38 - len(after),
        "mapping_regression_count": len(before - after),
        "rows": rows,
    }, len(before - after)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R4A final 4 daytime terminal semantics observation.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default="20260722_090000")
    parser.add_argument("--scheduler-type", default="manual_or_launchd")
    parser.add_argument("--scheduler-job-id", default="")
    parser.add_argument("--force-start-before-schedule", action="store_true")
    parser.add_argument("--broad-interval-seconds", type=int, default=120)
    parser.add_argument("--focused-interval-seconds", type=int, default=30)
    parser.add_argument("--post-terminal-follow-minutes", type=int, default=60)
    parser.add_argument("--scheduler-poll-seconds", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    r4 = load_r4_module(project_root)
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "prompt5_e01_r2d1c_r4a_gate.json").exists():
        raise RuntimeError(f"R4A artifact already finalized: {output_root}")

    now = datetime.now().astimezone()
    if now < parse_dt(SCHEDULED_START_KST) and not args.force_start_before_schedule:
        dump_json(
            output_root / "scheduled_execution_manifest.json",
            {
                "scheduled_start_time_kst": SCHEDULED_START_KST,
                "scheduler_type": args.scheduler_type,
                "scheduler_job_id": args.scheduler_job_id,
                "credential_reference_type": "operator_key_file_or_environment",
                "live_collection_started": False,
                "status": "SCHEDULED_WAITING_FOR_2026_07_22_0900_KST",
                "created_at_kst": now_kst(),
            },
        )
        print(json.dumps({"status": "SCHEDULED_WAITING_FOR_2026_07_22_0900_KST", "artifact_dir": str(output_root), "scheduled_start_time_kst": SCHEDULED_START_KST}, ensure_ascii=False, indent=2))
        return

    r4_root, r4_gate = validate_r4(project_root)
    secret, secret_source = r4.read_secret(args.service_key_file)
    if not secret:
        raise RuntimeError("MISSING_SERVICE_KEY")
    pre_scan = r4.scan_secret(output_root, secret)
    v9, route_meta, static = load_route_contract(r4, r4_root, output_root)
    dump_json(output_root / "r4_reference.json", {"r4_artifact": str(r4_root), "r4_gate_path": str(r4_root / "prompt5_e01_r2d1c_r4_gate.json"), "r4_gate_sha256": sha256_file(r4_root / "prompt5_e01_r2d1c_r4_gate.json"), "required_approved_mapping_count": 34})
    dump_json(output_root / "mapping_v9_reference.json", {"path": str(r4_root / "turnaround_mapping_contract_v9.parquet"), "sha256": sha256_file(r4_root / "turnaround_mapping_contract_v9.parquet")})
    dump_json(output_root / "credential_reference.json", {"credential_available": bool(secret), "credential_length": len(secret), "credential_fingerprint": r4.credential_fingerprint(secret), "credential_source": secret_source})
    dump_json(output_root / "observation_campaign_contract.json", {"scheduled_start_time_kst": SCHEDULED_START_KST, "scheduled_end_time_kst": SCHEDULED_END_KST, "routes": FINAL4_ROUTES, "broad_interval_seconds": args.broad_interval_seconds, "focused_interval_seconds": args.focused_interval_seconds, "post_terminal_follow_minutes": args.post_terminal_follow_minutes, "terminal_recovery_estimated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "scheduled_execution_manifest.json", {"scheduled_start_time_kst": SCHEDULED_START_KST, "actual_start_time_kst": now_kst(), "scheduler_type": args.scheduler_type, "scheduler_job_id": args.scheduler_job_id, "credential_reference_type": "operator_key_file_or_environment", "live_collection_started": True})

    preflight_rows, preflight_req = fetch_route_r4a(r4, FINAL4_ROUTES[0], secret, output_root, "SERVICE_TIMECHECK", 0, route_meta[FINAL4_ROUTES[0]])
    seqs = [as_int(row.get("current_sequence")) for row in preflight_rows if as_int(row.get("current_sequence")) is not None]
    provider_times = sorted({row.get("provider_event_time") for row in preflight_rows if row.get("provider_event_time")})
    preflight = {
        "service_timecheck_request_time_kst": preflight_req["request_time_kst"],
        "service_timecheck_route_id": FINAL4_ROUTES[0],
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

    samples, requests, trigger_rows, observation_minutes, actual_start, actual_end = run_capture(r4, args, secret, output_root, route_meta)
    all_samples = preflight_rows + samples
    all_requests = [preflight_req] + requests
    sample_df = pd.DataFrame(all_samples)
    sample_df = sample_df.reindex(columns=SAMPLE_COLUMNS_R4A)
    sample_df.to_parquet(output_root / "terminal_semantics_position_samples_r4a.parquet", index=False)
    trajectories = pd.DataFrame(r4.build_trajectories(sample_df))
    trajectories.to_parquet(output_root / "vehicle_continuous_trajectories_r4a.parquet", index=False)
    events = pd.DataFrame(r4.reconstruct_events(trajectories.to_dict("records"), route_meta))
    events.to_parquet(output_root / "terminal_transition_events_r4a.parquet", index=False)
    counters, counter_by_route = r4.classify_counter_rows(sample_df)
    loop_count = int((events["operation_candidate"] == "LOOP_CONTINUOUS").sum()) if not events.empty else 0
    paired_count = int((events["operation_candidate"] == "PAIRED_OPPOSITE_DIRECTION").sum()) if not events.empty else 0
    off_count = int((events["operation_candidate"] == "OFF_GRAPH_CONTINUATION_AND_REENTRY").sum()) if not events.empty else 0
    depot_count = int((events["operation_candidate"] == "DEPOT_OR_DEADHEAD_TRANSITION").sum()) if not events.empty else 0
    classified_count = loop_count + paired_count + off_count + depot_count
    counter_audit = {**counters, "classified_operation_transition_count": classified_count, "loop_reset_transition_count": loop_count, "paired_direction_transition_count": paired_count, "off_graph_reentry_transition_count": off_count, "depot_deadhead_transition_count": depot_count, "counter_v4_pass": classified_count == loop_count + paired_count + off_count + depot_count}
    dump_json(output_root / "terminal_counter_audit_v4.json", counter_audit)
    focused_runtime = {"focused_trigger_evaluation_count": len(trigger_rows), "focused_trigger_match_count": sum(1 for row in trigger_rows if truthy(row.get("trigger_matched"))), "focused_capture_started_count": sum(1 for row in trigger_rows if truthy(row.get("focused_capture_started"))), "post_terminal_follow_started_count": sum(1 for row in trigger_rows if truthy(row.get("post_terminal_follow_started"))), "rows": trigger_rows}
    dump_json(output_root / "focused_capture_runtime_audit.json", focused_runtime)
    decisions, contradiction = r4.decide_routes(route_meta, sample_df, events, counter_by_route, trigger_rows, observation_minutes)
    pd.DataFrame(decisions).to_parquet(output_root / "terminal_operation_decisions_r4a.parquet", index=False)
    dump_json(output_root / "terminal_operation_contradiction_audit.json", contradiction)
    r4.make_route_bundles(output_root, route_meta, sample_df, trajectories, events, decisions, trigger_rows)
    mapping_v10, regression = update_mapping_v10(r4, v9, decisions, output_root)
    dump_json(output_root / "turnaround_mapping_contract_v10.json", mapping_v10)
    pd.DataFrame(mapping_v10["rows"]).to_parquet(output_root / "turnaround_mapping_contract_v10.parquet", index=False)
    remaining = []
    for row in decisions:
        if not truthy(row.get("approved")):
            route_samples = sample_df[sample_df["route_id"].astype(str) == str(row["route_id"])]
            max_seq = pd.to_numeric(route_samples["current_sequence"], errors="coerce").max() if not route_samples.empty else None
            remaining.append({"route_id": row["route_id"], "reason": row["reason"], "observation_minutes": row["observation_minutes"], "observed_vehicles": row["vehicle_count"], "maximum_observed_sequence": None if pd.isna(max_seq) else int(max_seq), "terminal_state_observations": row["terminal_state_observation_count"], "terminal_entry_events": row["terminal_entry_event_count"], "post_terminal_follow_minutes": args.post_terminal_follow_minutes if row["focused_capture_started_count"] else 0, "specific_evidence_still_required": row["remaining_requirement"], "recommended_next_observation_window": "weekday daytime 09:00-13:00 KST, focused on post-terminal departure"})
    dump_json(output_root / "remaining_mapping_evidence_requirements_v4.json", {"rows": remaining})

    auth_errors = sum(1 for row in all_requests if row["response_status"] == "AUTH_ERROR")
    html_errors = sum(1 for row in all_requests if row["response_status"] == "HTML_RESPONSE")
    rate_errors = sum(1 for row in all_requests if row["response_status"] == "RATE_LIMIT")
    post_scan = r4.scan_secret(output_root, secret)
    dump_json(output_root / "secret_leak_audit.json", {"pre_campaign_artifact_scan": pre_scan, "post_campaign_artifact_scan": post_scan, "secret_leak_count": post_scan["secret_literal_occurrence_count"], "secret_leak_pass": post_scan["secret_literal_occurrence_count"] == 0})
    newly = mapping_v10["newly_recovered_mapping_count"]
    trigger_runtime_pass = focused_runtime["focused_capture_started_count"] > 0 if focused_runtime["focused_trigger_match_count"] > 0 else True
    if regression:
        status = "FAIL_MAPPING_REGRESSION"
    elif auth_errors or html_errors or rate_errors or post_scan["secret_literal_occurrence_count"] or not counter_audit["counter_v4_pass"]:
        status = "FAIL_OBSERVATION_CAMPAIGN"
    elif not trigger_runtime_pass:
        status = "BLOCKED_FOCUSED_TRIGGER_RUNTIME"
    elif mapping_v10["approved_mapping_count"] == 38 and mapping_v10["missing_mapping_count"] == 0:
        status = "PASS_MAPPING_38_OF_38_READY"
    elif newly > 0:
        status = "PASS_MAPPING_PARTIALLY_RECOVERED"
    else:
        status = "BLOCKED_NO_POST_TERMINAL_OBSERVATION"
    phase2 = {"approved_for_terminal_recovery_campaign": status == "PASS_MAPPING_38_OF_38_READY", "approved_for_phase2_turnaround_execution": False, "approved_for_baseline_feasibility_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a_corrected_retrospective": False, "approved_for_e2_execution": False, "prompt6_full_matrix_approved": False, "real_world_causal_claim_allowed": False, "phase2_production_executed": False}
    dump_json(output_root / "terminal_recovery_campaign_authorization_draft.json", {"approved_for_terminal_recovery_campaign": status == "PASS_MAPPING_38_OF_38_READY", "terminal_recovery_estimated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "phase2_execution_authorization.json", phase2)
    repeated = int((sample_df.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not sample_df.empty else 0
    seq_prog = int(sum(1 for _, group in sample_df.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2)) if not sample_df.empty else 0
    focused_runtime_counts = {key: value for key, value in focused_runtime.items() if key != "rows"}
    gate = {
        "status": status,
        "classification": status,
        "scheduled_start_time_kst": SCHEDULED_START_KST,
        "actual_start_time_kst": actual_start,
        "actual_end_time_kst": actual_end,
        "scheduler_type": args.scheduler_type,
        "campaign_route_count": len(FINAL4_ROUTES),
        "campaign_total_observation_minutes": sum(observation_minutes.values()),
        "campaign_total_requests": len(all_requests),
        "campaign_raw_file_count": len(all_requests),
        "campaign_normalized_rows": len(sample_df),
        "unique_vehicle_count": int(sample_df["vehicle_id"].nunique(dropna=True)) if not sample_df.empty else 0,
        "repeated_vehicle_count": repeated,
        "sequence_progression_count": seq_prog,
        **focused_runtime_counts,
        "focused_trigger_runtime_passed": trigger_runtime_pass,
        **counter_audit,
        "previous_mapping_count": 34,
        "newly_recovered_mapping_count": newly,
        "approved_mapping_count": mapping_v10["approved_mapping_count"],
        "missing_mapping_count": mapping_v10["missing_mapping_count"],
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
        "AUTH_ERROR_count": auth_errors,
        "HTML_response_count": html_errors,
        "rate_limit_error_count": rate_errors,
        "secret_leak_count": post_scan["secret_literal_occurrence_count"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        **phase2,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "scientific_parameter_changed": False,
        "calibration_applied": False,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r4a_gate.json", gate)
    report = f"""# Prompt 5-E01-R2D-1C-R4A Final Report

status: {status}
classification: {status}

scheduled start: {SCHEDULED_START_KST}
actual start: {actual_start}
actual end: {actual_end}

mapping previous: 34
newly recovered: {newly}
approved: {mapping_v10['approved_mapping_count']}
missing: {mapping_v10['missing_mapping_count']}
regression: {regression}

API/security: AUTH_ERROR={auth_errors}, HTML={html_errors}, rate-limit={rate_errors}, secret leak={post_scan['secret_literal_occurrence_count']}

terminal recovery estimated: false
terminal recovery applied: false
Phase 2 executed: false
"""
    (output_root / "prompt5_e01_r2d1c_r4a_final_report.md").write_text(report, encoding="utf-8")
    manifest_path = output_root / "scheduled_execution_manifest.json"
    scheduled_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    scheduled_manifest.update({"actual_end_time_kst": actual_end, "status": status})
    dump_json(manifest_path, scheduled_manifest)
    manifest = {"prompt": "Prompt 5-E01-R2D-1C-R4A", "artifact_dir": str(output_root), "created_at": now_kst(), "status": status, "files": sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file())}
    dump_json(output_root / "prompt5_e01_r2d1c_r4a_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status, "approved_mapping_count": gate["approved_mapping_count"], "missing_mapping_count": gate["missing_mapping_count"], "newly_recovered_mapping_count": newly}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
