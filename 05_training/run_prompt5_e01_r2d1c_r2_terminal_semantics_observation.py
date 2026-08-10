from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r2_terminal_semantics_observation"
R2D1A_ROOT = "05_training/artifacts/prompt5_e01_r2d1a_instrumentation_mapping_recovery_20260720_000000"
R2D1C_R1_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight_20260721_030000"
CAPTURE_SCRIPT = "05_training/data_acquisition/capture_terminal_semantics_samples.py"
MISSING_ROUTE_IDS = [
    "2000002000",
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
CANARY_ROUTES = ["3000232000", "2000002000"]
KEY_VALUE_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")
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
    "sequence_delta",
    "elapsed_seconds",
    "spatial_distance_m",
    "route_changed",
    "direction_changed",
    "sequence_reset",
    "vehicle_disappeared_then_reappeared",
    "batch_id",
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
    "operation_candidate",
    "classification",
    "source_raw_sha256s",
]


def kst_now() -> str:
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


def read_text_maybe_rtf(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        result = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], check=True, capture_output=True, text=True)
        return result.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def read_service_key(key_file: str) -> Tuple[str, str]:
    if key_file:
        text = read_text_maybe_rtf(Path(key_file).expanduser())
        match = KEY_VALUE_RE.search(text)
        if match:
            return match.group(1).strip(), "file"
        candidates = re.findall(r"[A-Za-z0-9%+/=_\\-]{20,}", text)
        if candidates:
            return max(candidates, key=len).strip(), "file"
    env_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    return env_key, "environment" if env_key else "missing"


def credential_fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def scan_for_secret(root: Path, secret: str) -> Dict[str, Any]:
    needles = {secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")}
    findings = []
    skip_names = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in skip_names for part in path.parts):
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except Exception:
            continue
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def validate_r1(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R2D1C_R1_ROOT
    gate_path = root / "prompt5_e01_r2d1c_r1_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = {
        "status": "BLOCKED_MAPPING_SEMANTICS_UNRESOLVED",
        "endpoint_preflight_pass_count": 5,
        "auth_error_count": 0,
        "html_response_count": 0,
        "previous_approved_mapping_count": 25,
        "newly_recovered_mapping_count": 0,
        "approved_mapping_count": 25,
        "missing_mapping_count": 13,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    mismatches = {key: {"expected": expected, "actual": gate.get(key)} for key, expected in required.items() if gate.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"R2D-1C-R1 prerequisite failed: {mismatches}")
    return root, gate


def build_static_topology(r1_root: Path, output_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    bs = pd.read_parquet(r1_root / "getbs02_missing_route_sequences.parquet")
    links = pd.read_parquet(r1_root / "getlink02_missing_route_links.parquet")
    rows: List[Dict[str, Any]] = []
    terminal_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for route_id in MISSING_ROUTE_IDS:
        route_bs = bs[bs["route_id"].astype(str) == route_id].copy()
        route_bs["stop_order_num"] = pd.to_numeric(route_bs["stop_order"], errors="coerce")
        for direction_id, group in route_bs.groupby(route_bs["direction_id"].astype(str), dropna=False):
            group = group.sort_values("stop_order_num")
            if group.empty:
                continue
            first = group.iloc[0]
            last = group.iloc[-1]
            terminal_by_key[(route_id, str(direction_id))] = {
                "route_id": route_id,
                "direction_id": str(direction_id),
                "min_sequence": int(first["stop_order_num"]) if pd.notna(first["stop_order_num"]) else None,
                "max_sequence": int(last["stop_order_num"]) if pd.notna(last["stop_order_num"]) else None,
                "first_stop_id": str(first["stop_id"]),
                "last_stop_id": str(last["stop_id"]),
                "terminal_candidate_stop_id": str(last["stop_id"]),
                "static_source_paths": [str(r1_root / "getbs02_missing_route_sequences.parquet"), str(r1_root / "getlink02_missing_route_links.parquet")],
                "static_source_sha256s": [sha256_file(r1_root / "getbs02_missing_route_sequences.parquet"), sha256_file(r1_root / "getlink02_missing_route_links.parquet")],
            }
            for _, row in group.iterrows():
                rows.append(
                    {
                        "route_id": route_id,
                        "direction_id": str(direction_id),
                        "record_type": "STOP_SEQUENCE",
                        "sequence": None if pd.isna(row["stop_order_num"]) else int(row["stop_order_num"]),
                        "stop_id": row.get("stop_id"),
                        "stop_name": row.get("stop_name"),
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "link_id": None,
                        "link_start_node": None,
                        "link_end_node": None,
                        "link_distance_m": None,
                        "link_travel_time_seconds": None,
                        "terminal_candidate_stop_id": str(last["stop_id"]),
                        "static_source_path": row.get("raw_path"),
                        "static_source_sha256": row.get("raw_sha256"),
                    }
                )
        route_links = links[links["route_id"].astype(str) == route_id].copy()
        route_links["link_seq_num"] = pd.to_numeric(route_links["link_seq"], errors="coerce")
        for _, row in route_links.sort_values(["direction_id", "link_seq_num"]).iterrows():
            rows.append(
                {
                    "route_id": route_id,
                    "direction_id": str(row.get("direction_id")),
                    "record_type": "LINK_SEQUENCE",
                    "sequence": None if pd.isna(row["link_seq_num"]) else int(row["link_seq_num"]),
                    "stop_id": None,
                    "stop_name": None,
                    "x": None,
                    "y": None,
                    "link_id": row.get("link_id"),
                    "link_start_node": row.get("st_node"),
                    "link_end_node": row.get("ed_node"),
                    "link_distance_m": row.get("gis_dist"),
                    "link_travel_time_seconds": None,
                    "terminal_candidate_stop_id": None,
                    "static_source_path": row.get("raw_path"),
                    "static_source_sha256": row.get("raw_sha256"),
                }
            )
    df = pd.DataFrame(rows)
    write_parquet(output_root / "unresolved_route_static_topology.parquet", rows)
    duplicate_sequence_count = int(df.duplicated(["route_id", "direction_id", "record_type", "sequence"]).sum()) if not df.empty else 0
    non_mono = 0
    for _, group in df.dropna(subset=["sequence"]).groupby(["route_id", "direction_id", "record_type"]):
        seq = group["sequence"].astype(int).tolist()
        if seq != sorted(seq):
            non_mono += 1
    null_identity = int(df["route_id"].isna().sum() + df["direction_id"].isna().sum()) if not df.empty else 0
    null_stop_link = int(
        df[(df["record_type"] == "STOP_SEQUENCE") & (df["stop_id"].isna() | (df["stop_id"].astype(str) == ""))].shape[0]
        + df[(df["record_type"] == "LINK_SEQUENCE") & (df["link_id"].isna() | (df["link_id"].astype(str) == ""))].shape[0]
    ) if not df.empty else 0
    audit = {
        "route_count": len(MISSING_ROUTE_IDS),
        "static_topology_row_count": len(rows),
        "duplicate_sequence_count": duplicate_sequence_count,
        "non_monotonic_sequence_count": non_mono,
        "null_route_id_count": null_identity,
        "null_stop_or_link_identity_count": null_stop_link,
        "link_travel_time_available": False,
        "static_topology_sufficient_as_sole_mapping_evidence": False,
    }
    dump_json(output_root / "unresolved_route_static_topology_audit.json", audit)
    return rows, audit, terminal_by_key


def build_batches() -> Dict[str, Any]:
    batches = [
        {"batch_id": "batch_1", "route_ids": ["2000002000", "3000232000", "4010002001", "4050001001"]},
        {"batch_id": "batch_2", "route_ids": ["2000002100", "3000323101", "4010002003", "4050010000"]},
        {"batch_id": "batch_3", "route_ids": ["2000003100", "3000410000", "3000410100", "4010002004", "4010002118"]},
    ]
    return {
        "timezone": "Asia/Seoul",
        "service_window": "05:00-22:00",
        "sampling_interval_seconds": 60,
        "max_calls_per_minute": 20,
        "batch_count": len(batches),
        "batches": batches,
        "full_campaign_executed": False,
        "canary_only_this_run": True,
    }


def run_capture(project_root: Path, output_root: Path, key_file: str, canary_seconds: int, interval: int, max_cpm: int) -> Tuple[Path, Dict[str, Any]]:
    capture_dir = output_root / "canary_capture"
    start = datetime.now().astimezone() + timedelta(seconds=2)
    end = start + timedelta(seconds=canary_seconds)
    cmd = [
        sys.executable,
        "-B",
        str(project_root / CAPTURE_SCRIPT),
        "--route-ids",
        *CANARY_ROUTES,
        "--start-time",
        start.isoformat(timespec="seconds"),
        "--end-time",
        end.isoformat(timespec="seconds"),
        "--interval-seconds",
        str(interval),
        "--output-dir",
        str(capture_dir),
        "--max-calls-per-minute",
        str(max_cpm),
        "--batch-id",
        "canary_20m",
        "--resume",
    ]
    if key_file:
        cmd.extend(["--service-key-file", key_file])
    result = subprocess.run(cmd, cwd=project_root, check=True, capture_output=True, text=True)
    capture_manifest = json.loads((capture_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    capture_manifest["capture_stdout_redacted"] = result.stdout.strip()
    return capture_dir, capture_manifest


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def haversine_like_m(x1: Any, y1: Any, x2: Any, y2: Any) -> Optional[float]:
    fx1, fy1, fx2, fy2 = to_float(x1), to_float(y1), to_float(x2), to_float(y2)
    if None in {fx1, fy1, fx2, fy2}:
        return None
    mean_lat = (fy1 + fy2) / 2.0
    dx = (fx2 - fx1) * 111_320.0 * max(0.1, __import__("math").cos(__import__("math").radians(mean_lat)))
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
            current = row.to_dict()
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
                "sequence_delta": None,
                "elapsed_seconds": None,
                "spatial_distance_m": None,
                "route_changed": False,
                "direction_changed": False,
                "sequence_reset": False,
                "vehicle_disappeared_then_reappeared": False,
                "batch_id": row.get("batch_id"),
                "raw_file_path": row.get("raw_file_path"),
                "raw_sha256": row.get("raw_sha256"),
            }
            if prev is not None:
                elapsed = (current["request_dt"] - prev["request_dt"]).total_seconds() if pd.notna(current["request_dt"]) and pd.notna(prev["request_dt"]) else None
                seq_delta = None
                if pd.notna(current["seq_num"]) and pd.notna(prev["seq_num"]):
                    seq_delta = float(current["seq_num"] - prev["seq_num"])
                out.update(
                    {
                        "previous_route_id": str(prev.get("route_id")),
                        "previous_direction_id": str(prev.get("direction_id")),
                        "previous_sequence": prev.get("current_sequence"),
                        "previous_stop_id": prev.get("current_stop_id"),
                        "previous_x": prev.get("x"),
                        "previous_y": prev.get("y"),
                        "previous_time": prev.get("request_time_kst"),
                        "sequence_delta": seq_delta,
                        "elapsed_seconds": elapsed,
                        "spatial_distance_m": haversine_like_m(prev.get("x"), prev.get("y"), row.get("x"), row.get("y")),
                        "route_changed": str(prev.get("route_id")) != str(row.get("route_id")),
                        "direction_changed": str(prev.get("direction_id")) != str(row.get("direction_id")),
                        "sequence_reset": bool(seq_delta is not None and seq_delta < -5),
                        "vehicle_disappeared_then_reappeared": bool(elapsed is not None and elapsed > 90),
                    }
                )
            rows.append(out)
            prev = current
    return rows


def reconstruct_events(trajectories: Sequence[Mapping[str, Any]], terminals: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    for row in trajectories:
        before_route = str(row.get("previous_route_id")) if row.get("previous_route_id") is not None else None
        before_dir = str(row.get("previous_direction_id")) if row.get("previous_direction_id") is not None else None
        if before_route is None or before_dir is None:
            continue
        terminal = terminals.get((before_route, before_dir))
        if terminal is None:
            continue
        prev_seq = to_float(row.get("previous_sequence"))
        max_seq = to_float(terminal.get("max_sequence"))
        prev_stop = str(row.get("previous_stop_id")) if row.get("previous_stop_id") is not None else None
        reached = bool((max_seq is not None and prev_seq is not None and prev_seq >= max_seq) or prev_stop == str(terminal.get("terminal_candidate_stop_id")))
        if not reached:
            continue
        elapsed = to_float(row.get("elapsed_seconds"))
        if elapsed is None or elapsed <= 0:
            continue
        after_route = str(row.get("route_id"))
        after_dir = str(row.get("direction_id"))
        after_seq = to_float(row.get("current_sequence"))
        min_seq = to_float(terminal.get("min_sequence"))
        operation = "UNRESOLVED_TERMINAL_OPERATION"
        if after_route == before_route and after_dir != before_dir and min_seq is not None and after_seq is not None and after_seq <= min_seq + 3:
            operation = "PAIRED_OPPOSITE_DIRECTION"
        elif after_route == before_route and after_dir == before_dir and bool(row.get("sequence_reset")) and min_seq is not None and after_seq is not None and after_seq <= min_seq + 3:
            operation = "LOOP_CONTINUOUS"
        elif after_route != before_route and bool(row.get("vehicle_disappeared_then_reappeared")):
            operation = "DEPOT_OR_DEADHEAD_TRANSITION"
        elif bool(row.get("vehicle_disappeared_then_reappeared")) and after_seq is not None and min_seq is not None and after_seq <= min_seq + 5:
            operation = "OFF_GRAPH_CONTINUATION_AND_REENTRY"
        events.append(
            {
                "route_id_before": before_route,
                "direction_id_before": before_dir,
                "vehicle_id": row.get("vehicle_id"),
                "terminal_stop_id": terminal.get("terminal_candidate_stop_id"),
                "terminal_sequence": terminal.get("max_sequence"),
                "terminal_reached_time_lower": row.get("previous_time"),
                "terminal_reached_time_upper": row.get("previous_time"),
                "post_terminal_time_lower": row.get("request_time_kst"),
                "post_terminal_time_upper": row.get("request_time_kst"),
                "route_id_after": after_route,
                "direction_id_after": after_dir,
                "sequence_after": row.get("current_sequence"),
                "stop_id_after": row.get("current_stop_id"),
                "elapsed_seconds_lower": elapsed,
                "elapsed_seconds_upper": elapsed + 60.0,
                "spatial_distance_m": row.get("spatial_distance_m"),
                "operation_candidate": operation,
                "classification": "OBSERVED_INTERVAL_CENSORED_TERMINAL_TRANSITION",
                "source_raw_sha256s": [row.get("raw_sha256")],
            }
        )
    return events


def decide_operations(samples: pd.DataFrame, events: pd.DataFrame, terminals: Mapping[Tuple[str, str], Mapping[str, Any]], output_root: Path, r1_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    decisions = []
    contradiction_count = 0
    multiple_patterns = 0
    for route_id in MISSING_ROUTE_IDS:
        route_terminals = [value for key, value in terminals.items() if key[0] == route_id]
        direction_id = str(route_terminals[0]["direction_id"]) if route_terminals else "1"
        route_samples = samples[samples["route_id"].astype(str) == route_id] if not samples.empty else pd.DataFrame()
        route_events = events[events["route_id_before"].astype(str) == route_id] if not events.empty else pd.DataFrame()
        terminal_reaches = 0
        if not route_samples.empty and route_terminals:
            max_seqs = {str(t["direction_id"]): to_float(t["max_sequence"]) for t in route_terminals}
            route_samples = route_samples.copy()
            route_samples["seq_num"] = pd.to_numeric(route_samples["current_sequence"], errors="coerce")
            for _, row in route_samples.iterrows():
                max_seq = max_seqs.get(str(row.get("direction_id")))
                if max_seq is not None and pd.notna(row["seq_num"]) and float(row["seq_num"]) >= max_seq:
                    terminal_reaches += 1
        op_counts = route_events["operation_candidate"].value_counts().to_dict() if not route_events.empty else {}
        supported_ops = {op: int(count) for op, count in op_counts.items() if op != "UNRESOLVED_TERMINAL_OPERATION"}
        contradictory = max(0, len(supported_ops) - 1)
        contradiction_count += contradictory
        if contradictory:
            multiple_patterns += 1
        decision = "UNRESOLVED_TERMINAL_OPERATION"
        confidence = "UNRESOLVED"
        approved = False
        reason = "NO_VEHICLE_OBSERVED"
        if not route_samples.empty:
            if terminal_reaches == 0:
                reason = "NO_TERMINAL_REACH_EVENT"
            elif route_events.empty:
                reason = "NO_POST_TERMINAL_OBSERVATION"
            elif contradictory:
                decision = "MULTIPLE_SERVICE_PATTERNS"
                reason = "MULTIPLE_SERVICE_PATTERNS"
            elif supported_ops:
                decision = max(supported_ops.items(), key=lambda item: item[1])[0]
                count = supported_ops[decision]
                vehicles = int(route_events[route_events["operation_candidate"] == decision]["vehicle_id"].nunique(dropna=True))
                confidence = "HIGH" if count >= 1 else "LOW"
                approved = confidence in {"HIGH", "MEDIUM"}
                reason = "static topology plus same-vehicle terminal transition" if approved else "LOW_CONFIDENCE_TRANSITION"
        vehicles = int(route_events["vehicle_id"].nunique(dropna=True)) if not route_events.empty else 0
        decision_row = {
            "route_id": route_id,
            "direction_id": direction_id,
            "decision": decision,
            "confidence": confidence,
            "supporting_transition_count": int(len(route_events[route_events["operation_candidate"] == decision])) if not route_events.empty and decision in set(route_events["operation_candidate"]) else 0,
            "independent_vehicle_count": vehicles,
            "contradictory_count": contradictory,
            "supporting_evidence": [],
            "counter_evidence": [],
            "reason": reason,
            "approved": approved,
        }
        decisions.append(decision_row)
        bundle = output_root / "mapping_evidence" / route_id
        bundle.mkdir(parents=True, exist_ok=True)
        route_static = {
            "route_id": route_id,
            "terminal_candidates": route_terminals,
            "static_source_paths": [str(r1_root / "getbs02_missing_route_sequences.parquet"), str(r1_root / "getlink02_missing_route_links.parquet")],
            "static_source_sha256s": [sha256_file(r1_root / "getbs02_missing_route_sequences.parquet"), sha256_file(r1_root / "getlink02_missing_route_links.parquet")],
        }
        dump_json(bundle / "route_static_topology.json", route_static)
        write_parquet(bundle / "vehicle_transition_events.parquet", route_events.to_dict("records") if not route_events.empty else [], columns=EVENT_COLUMNS)
        dump_json(
            bundle / "terminal_transition_summary.json",
            {
                "route_id": route_id,
                "normalized_sample_count": int(len(route_samples)),
                "terminal_reach_event_count": terminal_reaches,
                "terminal_transition_event_count": int(len(route_events)),
                "operation_counts": op_counts,
            },
        )
        dump_json(bundle / "operation_decision.json", decision_row)
        raw_paths = sorted(set(route_samples["raw_file_path"].dropna().astype(str).tolist())) if not route_samples.empty else []
        dump_json(bundle / "raw_file_index.json", {"route_id": route_id, "raw_file_count": len(raw_paths), "raw_files": [{"path": path, "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in raw_paths]})
    return decisions, {"contradictory_transition_count": contradiction_count, "multiple_service_pattern_route_count": multiple_patterns}


def build_mapping_v7(r1_root: Path, decisions: Sequence[Mapping[str, Any]], output_root: Path) -> Dict[str, Any]:
    v6 = pd.read_parquet(r1_root / "turnaround_mapping_contract_v6.parquet")
    decisions_by_route = {str(row["route_id"]): row for row in decisions}
    rows = []
    for _, row in v6.iterrows():
        out = row.to_dict()
        route_id = str(out.get("route_id"))
        decision = decisions_by_route.get(route_id)
        if decision and not bool(out.get("approved")):
            approved = bool(decision["approved"])
            out.update(
                {
                    "branch_id": out.get("branch_id"),
                    "service_pattern_id": out.get("service_pattern_id") or f"{route_id}:{decision['direction_id']}",
                    "terminal_operation_type": decision["decision"],
                    "terminal_operation_resolved": approved,
                    "next_route_id": None,
                    "next_direction_id": None,
                    "next_trip_initial_stop_id": None,
                    "service_exit_stop_id": None,
                    "service_reentry_stop_id": None,
                    "transition_elapsed_seconds_lower": None,
                    "transition_elapsed_seconds_upper": None,
                    "transition_path_stop_ids": [],
                    "transition_path_link_ids": [],
                    "live_transition_event_count": decision["supporting_transition_count"],
                    "independent_vehicle_count": decision["independent_vehicle_count"],
                    "contradictory_transition_count": decision["contradictory_count"],
                    "static_source_paths": [],
                    "static_source_sha256s": [],
                    "live_evidence_paths": [str(output_root / "mapping_evidence" / route_id / "vehicle_transition_events.parquet")],
                    "live_evidence_sha256s": [sha256_file(output_root / "mapping_evidence" / route_id / "vehicle_transition_events.parquet")],
                    "mapping_confidence": decision["confidence"],
                    "approved": approved,
                    "newly_recovered_in_r2d1c_r2": approved,
                }
            )
        else:
            out.update(
                {
                    "terminal_operation_resolved": bool(out.get("approved")),
                    "next_route_id": out.get("route_id") if bool(out.get("approved")) else None,
                    "live_transition_event_count": 0,
                    "independent_vehicle_count": 0,
                    "contradictory_transition_count": 0,
                    "static_source_paths": [],
                    "static_source_sha256s": [],
                    "live_evidence_paths": [],
                    "live_evidence_sha256s": [],
                    "newly_recovered_in_r2d1c_r2": False,
                }
            )
        rows.append(out)
    approved_count = sum(1 for row in rows if bool(row.get("approved")))
    payload = {
        "contract_version": "turnaround_mapping_contract_v7",
        "expected_mapping_count": 38,
        "previous_mapping_count": 25,
        "newly_recovered_mapping_count": sum(1 for row in rows if bool(row.get("newly_recovered_in_r2d1c_r2"))),
        "approved_mapping_count": approved_count,
        "missing_mapping_count": 38 - approved_count,
        "rows": rows,
    }
    dump_json(output_root / "turnaround_mapping_contract_v7.json", payload)
    write_parquet(output_root / "turnaround_mapping_contract_v7.parquet", rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R2 terminal semantics observation campaign.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--canary-seconds", type=int, default=1200)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--max-calls-per-minute", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    service_key, key_source = read_service_key(args.service_key_file)
    if not service_key:
        raise SystemExit("service key missing")
    r1_root, r1_gate = validate_r1(project_root)
    r1_gate_path = r1_root / "prompt5_e01_r2d1c_r1_gate.json"
    dump_json(output_root / "r2d1c_r1_reference.json", {"path": str(r1_gate_path), "sha256": sha256_file(r1_gate_path), "gate": r1_gate})
    authority_files = [
        "counter_semantics_contract_v3.json",
        "passenger_event_time_contract_v1.json",
        "phase1_counter_audit_v3.json",
        "phase1_passenger_event_time_audit_v2.json",
    ]
    dump_json(
        output_root / "instrumentation_authority_reference.json",
        {
            "counter_authority": "R2D-1A",
            "passenger_event_time_authority": "R2D-1A",
            "files": [{"path": str(project_root / R2D1A_ROOT / name), "sha256": sha256_file(project_root / R2D1A_ROOT / name)} for name in authority_files],
        },
    )
    dump_json(
        output_root / "credential_reference.json",
        {
            "credential_available": True,
            "credential_source": key_source,
            "credential_fingerprint": credential_fingerprint(service_key),
            "credential_length": len(service_key),
            "service_key_literal_recorded": False,
        },
    )
    pre_scan = scan_for_secret(output_root, service_key)

    static_rows, static_audit, terminal_by_key = build_static_topology(r1_root, output_root)
    batch_schedule = build_batches()
    dump_json(output_root / "route_batch_schedule.json", batch_schedule)
    dump_json(
        output_root / "observation_campaign_contract.json",
        {
            "purpose": "terminal operation semantics only; terminal recovery seconds are not estimated",
            "target_route_count": 13,
            "canary_route_ids": CANARY_ROUTES,
            "canary_seconds": args.canary_seconds,
            "full_campaign_executed": False,
            "service_window": "05:00-22:00 Asia/Seoul",
            "sampling_interval_seconds": args.interval_seconds,
            "max_calls_per_minute": args.max_calls_per_minute,
            "completion_rule": "HIGH/MEDIUM mapping evidence or full service window with evidence-specific blocker",
        },
    )
    capture_dir, capture_manifest = run_capture(project_root, output_root, args.service_key_file, args.canary_seconds, args.interval_seconds, args.max_calls_per_minute)
    samples_path = capture_dir / "terminal_semantics_position_samples.parquet"
    samples = pd.read_parquet(samples_path)
    shutil.copy2(samples_path, output_root / "terminal_semantics_position_samples.parquet")
    raw_file_count = len([request for request in capture_manifest.get("requests", [])])
    canary_pass = bool(
        raw_file_count > 0
        and len(samples) > 0
        and samples["vehicle_id"].nunique(dropna=True) > 0
        and (samples.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum() > 0
        and sum(1 for _, group in samples.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2) > 0
        and sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "AUTH_ERROR") == 0
        and sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "HTML_RESPONSE") == 0
        and sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "RATE_LIMIT") == 0
    )
    dump_json(
        output_root / "campaign_preflight_canary.json",
        {
            "canary_executed": True,
            "canary_route_ids": CANARY_ROUTES,
            "canary_seconds": args.canary_seconds,
            "raw_file_count": raw_file_count,
            "normalized_row_count": int(len(samples)),
            "unique_vehicle_count": int(samples["vehicle_id"].nunique(dropna=True)) if not samples.empty else 0,
            "same_vehicle_repeated_observation_count": int((samples.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not samples.empty else 0,
            "sequence_progression_count": int(sum(1 for _, group in samples.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2)) if not samples.empty else 0,
            "auth_error_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "AUTH_ERROR"),
            "html_response_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "HTML_RESPONSE"),
            "rate_limit_error_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "RATE_LIMIT"),
            "pass": canary_pass,
        },
    )

    trajectories = build_trajectories(samples)
    write_parquet(output_root / "vehicle_continuous_trajectories.parquet", trajectories, columns=TRAJECTORY_COLUMNS)
    events = reconstruct_events(trajectories, terminal_by_key)
    write_parquet(output_root / "terminal_transition_events.parquet", events, columns=EVENT_COLUMNS)
    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    negative_elapsed = int((pd.to_numeric(events_df["elapsed_seconds_lower"], errors="coerce") < 0).sum()) if not events_df.empty else 0
    terminal_reach_event_count = 0
    if not samples.empty:
        samples_for_reach = samples.copy()
        samples_for_reach["seq_num"] = pd.to_numeric(samples_for_reach["current_sequence"], errors="coerce")
        for _, row in samples_for_reach.iterrows():
            terminal = terminal_by_key.get((str(row.get("route_id")), str(row.get("direction_id"))))
            if terminal and pd.notna(row["seq_num"]) and float(row["seq_num"]) >= float(terminal["max_sequence"]):
                terminal_reach_event_count += 1
    transition_audit = {
        "terminal_transition_event_count": len(events),
        "terminal_reach_event_count": terminal_reach_event_count,
        "post_terminal_observation_count": len(events),
        "negative_elapsed_count": negative_elapsed,
        "terminal_transition_event_invariants_pass": negative_elapsed == 0,
        "exact_transition_time_claim_allowed": False,
    }
    dump_json(output_root / "terminal_transition_event_audit.json", transition_audit)

    decisions, contradiction_audit = decide_operations(samples, events_df, terminal_by_key, output_root, r1_root)
    write_parquet(output_root / "terminal_operation_decisions.parquet", decisions)
    dump_json(output_root / "terminal_operation_contradiction_audit.json", contradiction_audit)
    mapping_v7 = build_mapping_v7(r1_root, decisions, output_root)

    remaining = []
    for decision in decisions:
        if not decision["approved"]:
            remaining.append(
                {
                    "route_id": decision["route_id"],
                    "direction_id": decision["direction_id"],
                    "blocker": decision["reason"],
                    "required_additional_evidence": "same vehicle terminal reach plus post-terminal first observation",
                    "recommended_sampling_duration": "one continuous service window or until >=3 terminal transitions",
                    "recommended_sampling_interval_seconds": 60,
                    "specific_route_branch_source_required": "BIS getPos02 repeated observations plus static getBs02/getLink02 support",
                }
            )
    dump_json(output_root / "remaining_mapping_evidence_requirements.json", {"rows": remaining})
    recovery_auth = {
        "approved_for_terminal_recovery_campaign": mapping_v7["approved_mapping_count"] == 38,
        "terminal_recovery_seconds": "unresolved",
        "terminal_recovery_values_estimated": False,
        "terminal_recovery_applied": False,
        "reason": "Terminal recovery campaign requires mapping 38/38 first.",
    }
    dump_json(output_root / "terminal_recovery_campaign_authorization_draft.json", recovery_auth)
    phase2_auth = {
        "approved_for_terminal_recovery_campaign": recovery_auth["approved_for_terminal_recovery_campaign"],
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "phase2_production_executed": False,
    }
    dump_json(output_root / "phase2_execution_authorization.json", phase2_auth)

    post_scan = scan_for_secret(output_root, service_key)
    dump_json(output_root / "secret_leak_audit.json", {"pre_campaign_output_scan": pre_scan, "post_campaign_artifact_scan": post_scan, "secret_leak_count": post_scan["secret_literal_occurrence_count"], "secret_leak_pass": post_scan["secret_literal_occurrence_count"] == 0})
    newly = mapping_v7["newly_recovered_mapping_count"]
    if post_scan["secret_literal_occurrence_count"] > 0 or not canary_pass or negative_elapsed > 0:
        status = "FAIL_OBSERVATION_CAMPAIGN" if post_scan["secret_literal_occurrence_count"] == 0 else "FAIL_OBSERVATION_CAMPAIGN_SECRET_LEAK"
    elif mapping_v7["approved_mapping_count"] == 38:
        status = "PASS_MAPPING_38_OF_38_READY"
    elif newly > 0:
        status = "PASS_MAPPING_PARTIALLY_RECOVERED"
    elif contradiction_audit["multiple_service_pattern_route_count"] > 0:
        status = "BLOCKED_MULTIPLE_SERVICE_PATTERNS"
    else:
        status = "BLOCKED_INSUFFICIENT_TERMINAL_TRANSITIONS"

    op_counts = pd.DataFrame(decisions)["decision"].value_counts().to_dict() if decisions else {}
    confidence_counts = pd.DataFrame(decisions)["confidence"].value_counts().to_dict() if decisions else {}
    repeated_vehicle_count = int((samples.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not samples.empty else 0
    sequence_progression_count = int(sum(1 for _, group in samples.groupby(["route_id", "direction_id", "vehicle_id"]) if pd.to_numeric(group["current_sequence"], errors="coerce").nunique(dropna=True) >= 2)) if not samples.empty else 0
    gate = {
        "status": status,
        "classification": status,
        "r2d1c_r1_gate_path": str(r1_gate_path),
        "r2d1c_r1_gate_sha256": sha256_file(r1_gate_path),
        "campaign_route_count": len(CANARY_ROUTES),
        "campaign_batch_count": 1,
        "campaign_service_day_count": 1,
        "campaign_total_requests": raw_file_count,
        "campaign_raw_file_count": raw_file_count,
        "campaign_normalized_rows": int(len(samples)),
        "unique_vehicle_count": int(samples["vehicle_id"].nunique(dropna=True)) if not samples.empty else 0,
        "repeated_vehicle_count": repeated_vehicle_count,
        "sequence_progression_count": sequence_progression_count,
        "terminal_reach_event_count": terminal_reach_event_count,
        "post_terminal_observation_count": len(events),
        "terminal_transition_event_count": len(events),
        "previous_mapping_count": 25,
        "newly_recovered_mapping_count": newly,
        "approved_mapping_count": mapping_v7["approved_mapping_count"],
        "missing_mapping_count": mapping_v7["missing_mapping_count"],
        "paired_opposite_direction_count": op_counts.get("PAIRED_OPPOSITE_DIRECTION", 0),
        "loop_continuous_count": op_counts.get("LOOP_CONTINUOUS", 0),
        "off_graph_reentry_count": op_counts.get("OFF_GRAPH_CONTINUATION_AND_REENTRY", 0),
        "depot_deadhead_count": op_counts.get("DEPOT_OR_DEADHEAD_TRANSITION", 0),
        "multiple_service_pattern_count": op_counts.get("MULTIPLE_SERVICE_PATTERNS", 0),
        "unresolved_terminal_operation_count": op_counts.get("UNRESOLVED_TERMINAL_OPERATION", 0),
        "high_confidence_mapping_count": confidence_counts.get("HIGH", 0),
        "medium_confidence_mapping_count": confidence_counts.get("MEDIUM", 0),
        "low_confidence_mapping_count": confidence_counts.get("LOW", 0),
        "contradictory_transition_count": contradiction_audit["contradictory_transition_count"],
        "AUTH_ERROR_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "AUTH_ERROR"),
        "HTML_response_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "HTML_RESPONSE"),
        "rate_limit_error_count": sum(1 for request in capture_manifest.get("requests", []) if request.get("response_status") == "RATE_LIMIT"),
        "secret_leak_count": post_scan["secret_literal_occurrence_count"],
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
        "calibration_applied": False,
        "scientific_parameter_changed": False,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r2_gate.json", gate)
    report = f"""# Prompt 5-E01-R2D-1C-R2 Final Report

[Prompt 5-E01-R2D-1C-R2 판정]
status: {status}
classification: {status}

[Campaign]
routes: {len(CANARY_ROUTES)} canary routes of 13 planned
batches: 1 canary batch, 3 full batches planned
service days: 1 partial day
requests: {raw_file_count}
raw files: {raw_file_count}
normalized rows: {len(samples)}

[Vehicle continuity]
unique vehicles: {gate["unique_vehicle_count"]}
repeated vehicles: {repeated_vehicle_count}
sequence progression: {sequence_progression_count}
terminal reaches: {terminal_reach_event_count}
post-terminal observations: {len(events)}
valid transitions: {len(events)}

[Mapping]
previous: 25
newly recovered: {newly}
approved: {mapping_v7["approved_mapping_count"]}
missing: {mapping_v7["missing_mapping_count"]}

paired direction: {gate["paired_opposite_direction_count"]}
loop: {gate["loop_continuous_count"]}
off-graph re-entry: {gate["off_graph_reentry_count"]}
depot/deadhead: {gate["depot_deadhead_count"]}
multiple patterns: {gate["multiple_service_pattern_count"]}
unresolved: {gate["unresolved_terminal_operation_count"]}

[Confidence]
HIGH: {gate["high_confidence_mapping_count"]}
MEDIUM: {gate["medium_confidence_mapping_count"]}
LOW: {gate["low_confidence_mapping_count"]}
contradictions: {gate["contradictory_transition_count"]}

[Security/API]
AUTH_ERROR: {gate["AUTH_ERROR_count"]}
HTML: {gate["HTML_response_count"]}
rate-limit: {gate["rate_limit_error_count"]}
secret leak: {gate["secret_leak_count"]}

[Recovery]
recovery values estimated: false
recovery applied: false
campaign approved: {str(gate["approved_for_terminal_recovery_campaign"]).lower()}

[Guard]
test split: false
test target: false
test embedding: false
threshold changed: false
Phase 2 executed: false

[Next gate]
additional terminal observation: true
terminal recovery campaign: {str(gate["approved_for_terminal_recovery_campaign"]).lower()}
Phase 2: false
baseline feasibility: false
E0/E1 retraining: false
Prompt 6A: false
E2: false
"""
    (output_root / "prompt5_e01_r2d1c_r2_final_report.md").write_text(report, encoding="utf-8")
    dump_json(
        output_root / "prompt5_e01_r2d1c_r2_manifest.json",
        {
            "artifact_version": "prompt5_e01_r2d1c_r2_terminal_semantics_observation_v1",
            "created_at_kst": kst_now(),
            "artifact_dir": str(output_root),
            "files": sorted([str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()]),
        },
    )
    print(json.dumps({"artifact_dir": str(output_root), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
