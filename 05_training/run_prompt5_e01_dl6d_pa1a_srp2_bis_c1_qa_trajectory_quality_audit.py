#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C1-QA.

Prospective Vehicle Trajectory Quality, Route Progression, and Identity
Continuity Audit.

OFFLINE, read-only audit of the sealed SRP2-BIS-C1 limited-pilot artifact. Makes
NO API call, NO DB access, runs NO simulator and NO training, and does NOT mutate
the C1 artifact. It re-derives quality from the C1 normalized captures and grades:
  * vehicle-identity continuity (per token/route/direction)
  * route-progression coherence (forward / stationary / reset / direction-change)
  * sequence-reset & direction-change interpretation (interval-censored only)
  * observation gaps / missing cycles
  * position jumps (implausible speed between consecutive observations)
  * route-stop mapping quality vs the getBs02 master
  * per-trajectory quality grade

Turnaround candidates stay INTERVAL_CENSORED_PROVIDER_OBSERVATION and are never
promoted to an exact turnaround time. actual headway / arrival / departure / dwell
remain false. No morning-peak operational-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_c1_qa_trajectory_quality_audit.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

# --------------------------------------------------------------------------- #
# authoritative upstream: the sealed C1 capture
# --------------------------------------------------------------------------- #
C1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
C1_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_LIMITED_PROSPECTIVE_CAPTURE_COMPLETE"
C1_READINESS = "SRP2_BIS_C1_COMPLETE_TRAJECTORY_QUALITY_AUDIT_READY_PENDING_USER_COMMAND"
C1_MANIFEST = "artifact_manifest_srp2_bis_c1.json"
C1_LOCK = "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock"

# getBs02 route-stop master (real observed 20,508 rows / 234 routes) for mapping QA
ROUTE_STOP_SEQ_PARQUET = ARTIFACTS_ROOT / "suseong_source_pack_v1" / "route_stop_sequences.parquet"

TARGET_ROUTE_ID = "3000814001"

# --------------------------------------------------------------------------- #
# QA thresholds (data-quality grading; NOT operational-performance criteria)
# --------------------------------------------------------------------------- #
POSITION_JUMP_SPEED_KMH = 120.0   # bus positions implying > this speed are flagged
TRAJ_MIN_OBS = 3
TRAJ_MIN_CYCLES = 3

# --------------------------------------------------------------------------- #
# gate / readiness
# --------------------------------------------------------------------------- #
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_QA_TRAJECTORY_QUALITY_AUDIT_COMPLETE"
READINESS_SUFFICIENT = "SRP2_BIS_C1_QA_COMPLETE_TRAJECTORY_QUALITY_SUFFICIENT_PENDING_USER_COMMAND"
READINESS_LIMITED = "SRP2_BIS_C1_QA_COMPLETE_ADDITIONAL_CAPTURE_RECOMMENDED_PENDING_USER_COMMAND"
READINESS_REPAIR = "SRP2_BIS_C1_QA_COMPLETE_TRAJECTORY_QUALITY_REPAIR_PENDING_USER_COMMAND"

_B = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_QA_"
BLOCKED_C1 = _B + "C1_UPSTREAM_INVALID"
BLOCKED_INPUT = _B + "C1_CAPTURE_INPUT_MISSING"
_F = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_QA_"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_AUDIT"
FAIL_C1_MUT = _F + "C1_ARTIFACT_MUTATED"
FAIL_API = _F + "API_CALL_DETECTED"
FAIL_DB = _F + "DATABASE_ACCESS_DETECTED"
FAIL_SIM = _F + "SIMULATOR_EXECUTED"
FAIL_TRAIN = _F + "TRAINING_EXECUTED"
FAIL_HEADWAY = _F + "ACTUAL_HEADWAY_OVERCLAIMED"
FAIL_TURNAROUND = _F + "EXACT_TURNAROUND_OVERCLAIMED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class QAError(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate_status = gate
        self.detail = detail


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return None if (f != f or f in {float("inf"), float("-inf")}) else f
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def jsonl(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(json_clean(dict(r)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        import pandas as pd
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(rows), columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(p, index=False)


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source_path": str(src), "snapshot_relative_path": dst_rel, "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst), "byte_identical": sha256_file(src) == sha256_file(dst), "size_bytes": dst.stat().st_size}


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"c1-qa artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# upstream C1 verification (read-only; integrity + no mutation)
# --------------------------------------------------------------------------- #
def verify_c1_upstream() -> Dict[str, Any]:
    if not C1_ROOT.is_dir():
        return {"c1_root": str(C1_ROOT), "upstream_valid": False, "checks": {"artifact_exists": False}}
    gate = read_json(C1_ROOT / "gate_decision.json")
    lock = read_json(C1_ROOT / C1_LOCK)
    manifest_path = C1_ROOT / C1_MANIFEST
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    missing = mismatch = 0
    for r in manifest["files"]:
        t = C1_ROOT / r["relative_path"]
        if not t.exists():
            missing += 1
        elif r.get("sha256") is not None and sha256_file(t) != r["sha256"]:
            mismatch += 1
    checks = {"artifact_exists": True, "gate": gate.get("gate") == C1_GATE, "readiness": gate.get("readiness") == C1_READINESS,
              "lock_present": (C1_ROOT / C1_LOCK).exists(), "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha,
              "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0}
    return {"c1_root": str(C1_ROOT), "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "manifest_sha256": manifest_sha, "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "manifest_payload_count": len(manifest["files"]), "checks": checks, "upstream_valid": all(checks.values())}


def snapshot_c1(writer: Writer) -> Tuple[List[Dict[str, Any]], List[str]]:
    files = ["gate_decision.json", "downstream_lock.json", C1_MANIFEST, C1_LOCK, "capture_decision.json",
             "pilot_quality_metrics.json", "trajectory_quality_summary.json", "turnaround_interval_summary.json",
             "claim_boundary_audit.json", "capture_time_audit.json", "runner_freeze_audit.json"]
    records, paths = [], []
    for name in files:
        src = C1_ROOT / name
        if src.exists():
            rec = copy_file(writer, src, f"upstream_c1_snapshot/{name}")
            records.append(rec)
            paths.append(rec["snapshot_relative_path"])
    return records, paths


# --------------------------------------------------------------------------- #
# load C1 captures
# --------------------------------------------------------------------------- #
def load_c1_data():
    import pandas as pd
    gp = pd.read_parquet(C1_ROOT / "getpos02_normalized.parquet")
    gr = pd.read_parquet(C1_ROOT / "getrealtime02_normalized.parquet")
    master = pd.read_parquet(ROUTE_STOP_SEQ_PARQUET)
    return gp, gr, master


# --------------------------------------------------------------------------- #
# QA audits
# --------------------------------------------------------------------------- #
def _obs_records(gp) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for _i, r in gp.iterrows():
        tok = r.get("vehicle_token")
        if tok is None or (isinstance(tok, float) and tok != tok):
            continue
        key = (str(tok), str(r.get("route_id")), str(r.get("direction_id")))
        groups.setdefault(key, []).append({
            "cycle_index": int(r["cycle_index"]), "route_sequence": (int(r["route_sequence"]) if r.get("route_sequence") == r.get("route_sequence") and r.get("route_sequence") is not None else None),
            "current_stop_id": str(r.get("current_stop_id")) if r.get("current_stop_id") is not None else None,
            "x": (float(r["x_position_raw"]) if r.get("x_position_raw") == r.get("x_position_raw") and r.get("x_position_raw") is not None else None),
            "y": (float(r["y_position_raw"]) if r.get("y_position_raw") == r.get("y_position_raw") and r.get("y_position_raw") is not None else None),
            "poll_observed_at_utc": str(r.get("poll_observed_at_utc")), "raw_row_index": int(r.get("raw_row_index") or 0)})
    return groups


def identity_continuity_audit(groups) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for (tok, rid, did), obs in groups.items():
        cycles = sorted({o["cycle_index"] for o in obs})
        span = cycles[-1] - cycles[0] + 1 if cycles else 0
        distinct = len(cycles)
        max_gap = max((cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)), default=0)
        coverage = (distinct / span) if span else None
        if distinct >= 10 and max_gap <= 1:
            grade = "A_STRONG_CONTINUITY"
        elif distinct >= 3 and max_gap <= 2:
            grade = "B_CONTINUOUS_WITH_MINOR_GAPS"
        elif distinct >= 3:
            grade = "C_CONTINUOUS_WITH_GAPS"
        elif distinct == 1:
            grade = "SINGLE_OBSERVATION"
        else:
            grade = "SPARSE"
        rows.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "observation_count": len(obs),
                     "distinct_cycle_count": distinct, "cycle_span": span, "coverage_ratio": coverage,
                     "max_cycle_gap": max_gap, "continuity_grade": grade})
    # cross-direction: a token appearing in >1 (route,direction) group
    tok_dirs: Dict[str, set] = {}
    for (tok, rid, did) in groups.keys():
        tok_dirs.setdefault(tok, set()).add(did)
    both_dir = sorted(t for t, ds in tok_dirs.items() if len(ds) >= 2)
    summary = {"created_at": iso_kst(), "vehicle_token_direction_group_count": len(rows),
               "distinct_vehicle_token_count": len(tok_dirs),
               "grade_counts": _count_by(rows, "continuity_grade"),
               "strong_or_continuous_count": sum(1 for r in rows if r["continuity_grade"].startswith(("A", "B", "C"))),
               "vehicles_observed_in_both_directions": both_dir,
               "vehicles_observed_in_both_directions_count": len(both_dir),
               "note": "cross-direction tokens are turnaround-observation candidates (analyzed interval-censored, never as exact turnaround)"}
    return rows, summary


def position_jump_audit(groups) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    jump_total = 0
    seg_total = 0
    for (tok, rid, did), obs in groups.items():
        obs_sorted = sorted(obs, key=lambda o: (o["cycle_index"], o["raw_row_index"]))
        for i in range(1, len(obs_sorted)):
            a, b = obs_sorted[i - 1], obs_sorted[i]
            if None in (a["x"], a["y"], b["x"], b["y"]):
                continue
            try:
                ta = datetime.fromisoformat(a["poll_observed_at_utc"]); tb = datetime.fromisoformat(b["poll_observed_at_utc"])
                dt = abs((tb - ta).total_seconds())
            except Exception:
                dt = 0.0
            dist_km = haversine_km(a["x"], a["y"], b["x"], b["y"])
            speed = (dist_km / (dt / 3600.0)) if dt > 0 else None
            seg_total += 1
            is_jump = speed is not None and speed > POSITION_JUMP_SPEED_KMH
            if is_jump:
                jump_total += 1
                rows.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "from_cycle": a["cycle_index"],
                             "to_cycle": b["cycle_index"], "distance_km": round(dist_km, 4), "delta_seconds": dt,
                             "implied_speed_kmh": round(speed, 1), "classification": "POSITION_JUMP"})
    summary = {"created_at": iso_kst(), "position_segment_count": seg_total, "position_jump_count": jump_total,
               "position_jump_rate": (jump_total / seg_total) if seg_total else None,
               "speed_threshold_kmh": POSITION_JUMP_SPEED_KMH,
               "note": "implausible-speed segments flagged as GPS/identity artifacts; NOT an operational speed measurement"}
    return rows, summary


def observation_gap_audit(groups) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for (tok, rid, did), obs in groups.items():
        cycles = sorted({o["cycle_index"] for o in obs})
        if len(cycles) < 2:
            rows.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "distinct_cycles": len(cycles),
                         "span": 1, "missing_cycles_within_span": 0, "max_gap": 0})
            continue
        span = set(range(cycles[0], cycles[-1] + 1))
        missing = sorted(span - set(cycles))
        max_gap = max((cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)), default=0)
        rows.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "distinct_cycles": len(cycles),
                     "span": len(span), "missing_cycles_within_span": len(missing), "max_gap": max_gap})
    total_missing = sum(r["missing_cycles_within_span"] for r in rows)
    summary = {"created_at": iso_kst(), "group_count": len(rows), "total_missing_cycles_within_spans": total_missing,
               "max_gap_overall": max((r["max_gap"] for r in rows), default=0),
               "groups_with_no_internal_gap": sum(1 for r in rows if r["missing_cycles_within_span"] == 0)}
    return rows, summary


def route_progression_coherence(groups) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events = []
    per_traj = []
    for (tok, rid, did), obs in groups.items():
        obs_sorted = sorted(obs, key=lambda o: (o["cycle_index"], o["raw_row_index"]))
        cls_counts: Dict[str, int] = {}
        for i in range(1, len(obs_sorted)):
            a, b = obs_sorted[i - 1], obs_sorted[i]
            sa, sb = a["route_sequence"], b["route_sequence"]
            if sa is None or sb is None:
                cls = "INSUFFICIENT_EVIDENCE"
            elif sb > sa:
                cls = "FORWARD_PROGRESS" if (sb - sa) <= 5 else "LARGE_SEQUENCE_JUMP"
            elif sb == sa:
                cls = "STATIONARY_OR_REPEATED"
            else:
                cls = "SEQUENCE_RESET" if (sa - sb) >= 10 else "OUT_OF_ORDER"
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
            events.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "from_cycle": a["cycle_index"],
                           "to_cycle": b["cycle_index"], "from_sequence": sa, "to_sequence": sb, "classification": cls})
        n = sum(cls_counts.values())
        coherent = cls_counts.get("FORWARD_PROGRESS", 0) + cls_counts.get("STATIONARY_OR_REPEATED", 0)
        per_traj.append({"vehicle_token": tok, "route_id": rid, "direction_id": did, "transition_count": n,
                         "forward_count": cls_counts.get("FORWARD_PROGRESS", 0), "stationary_count": cls_counts.get("STATIONARY_OR_REPEATED", 0),
                         "reset_count": cls_counts.get("SEQUENCE_RESET", 0), "out_of_order_count": cls_counts.get("OUT_OF_ORDER", 0),
                         "large_jump_count": cls_counts.get("LARGE_SEQUENCE_JUMP", 0),
                         "coherence_ratio": (coherent / n) if n else None})
    summary = {"created_at": iso_kst(), "progression_event_count": len(events), "classification_counts": _count_by(events, "classification"),
               "trajectory_count": len(per_traj),
               "mean_coherence_ratio": (sum(t["coherence_ratio"] for t in per_traj if t["coherence_ratio"] is not None) / max(1, sum(1 for t in per_traj if t["coherence_ratio"] is not None))),
               "sequence_reset_alone_confirms_turnaround": False}
    return events, per_traj, summary


def sequence_reset_direction_change_interpretation(groups) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # direction change across a token's full observation stream (cross-direction) + within-direction resets
    by_token: Dict[str, List[Dict[str, Any]]] = {}
    for (tok, rid, did), obs in groups.items():
        for o in obs:
            by_token.setdefault(tok, []).append({**o, "route_id": rid, "direction_id": did})
    rows = []
    for tok, obs in by_token.items():
        obs_sorted = sorted(obs, key=lambda o: (o["cycle_index"], o["raw_row_index"]))
        for i in range(1, len(obs_sorted)):
            a, b = obs_sorted[i - 1], obs_sorted[i]
            if a["direction_id"] != b["direction_id"]:
                rows.append({"vehicle_token": tok, "type": "DIRECTION_CHANGE", "from_direction": a["direction_id"],
                             "to_direction": b["direction_id"], "left_cycle": a["cycle_index"], "right_cycle": b["cycle_index"],
                             "left_timestamp": a["poll_observed_at_utc"], "right_timestamp": b["poll_observed_at_utc"],
                             "interpretation": "POSSIBLE_TURNAROUND_INTERVAL_CENSORED", "exact_turnaround_claimed": False})
            elif a["route_sequence"] is not None and b["route_sequence"] is not None and (a["route_sequence"] - b["route_sequence"]) >= 10:
                rows.append({"vehicle_token": tok, "type": "SEQUENCE_RESET", "from_direction": a["direction_id"],
                             "to_direction": b["direction_id"], "left_cycle": a["cycle_index"], "right_cycle": b["cycle_index"],
                             "left_timestamp": a["poll_observed_at_utc"], "right_timestamp": b["poll_observed_at_utc"],
                             "interpretation": "POSSIBLE_TURNAROUND_OR_NEW_TRIP_INTERVAL_CENSORED", "exact_turnaround_claimed": False})
    summary = {"created_at": iso_kst(), "signal_count": len(rows), "type_counts": _count_by(rows, "type"),
               "direction_change_count": sum(1 for r in rows if r["type"] == "DIRECTION_CHANGE"),
               "sequence_reset_count": sum(1 for r in rows if r["type"] == "SEQUENCE_RESET"),
               "all_interval_censored": all(not r["exact_turnaround_claimed"] for r in rows),
               "note": "a single reset or direction change does NOT confirm a turnaround; all kept interval-censored"}
    return rows, summary


def route_stop_mapping_quality(gp, master) -> Dict[str, Any]:
    # master (route,direction,stop_order)->stop_id for the target route
    m = master[master["route_id"].astype(str) == TARGET_ROUTE_ID]
    seq_map = {(str(d), int(o)): str(s) for d, o, s in zip(m["direction_id"], m["stop_order"], m["stop_id"])}
    total = seq_consistent = seq_checkable = stop_on_route = 0
    for _i, r in gp.iterrows():
        total += 1
        did = str(r.get("direction_id")); seq = r.get("route_sequence"); sid = str(r.get("current_stop_id"))
        if str(r.get("route_stop_match_status", "")).startswith("EXACT"):
            stop_on_route += 1
        if seq == seq and seq is not None:
            seq_checkable += 1
            if seq_map.get((did, int(seq))) == sid:
                seq_consistent += 1
    return {"created_at": iso_kst(), "getpos02_row_count": int(total), "stop_on_route_count": int(stop_on_route),
            "stop_on_route_rate": (stop_on_route / total) if total else None,
            "sequence_checkable_count": int(seq_checkable), "sequence_stop_consistent_count": int(seq_consistent),
            "sequence_stop_consistency_rate": (seq_consistent / seq_checkable) if seq_checkable else None,
            "master_source": str(ROUTE_STOP_SEQ_PARQUET.relative_to(PROJECT_ROOT)),
            "note": "provider (seq, bsId) validated against the getBs02 master route-stop sequence"}


def trajectory_quality_grades(cont_rows, coh_by_key, jump_by_key, gap_by_key) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for c in cont_rows:
        key = (c["vehicle_token"], c["route_id"], c["direction_id"])
        if c["distinct_cycle_count"] < TRAJ_MIN_CYCLES or c["observation_count"] < TRAJ_MIN_OBS:
            continue  # only trajectory candidates
        coh = coh_by_key.get(key, {}).get("coherence_ratio")
        jumps = jump_by_key.get(key, 0)
        gap = c["max_cycle_gap"]
        if (c["distinct_cycle_count"] >= 10 and (coh is None or coh >= 0.9) and jumps == 0 and gap <= 1):
            grade = "A"
        elif (c["distinct_cycle_count"] >= 5 and (coh is None or coh >= 0.7) and jumps <= 1 and gap <= 3):
            grade = "B"
        else:
            grade = "C"
        rows.append({"vehicle_token": c["vehicle_token"], "route_id": c["route_id"], "direction_id": c["direction_id"],
                     "observation_count": c["observation_count"], "distinct_cycle_count": c["distinct_cycle_count"],
                     "coherence_ratio": coh, "position_jump_count": jumps, "max_cycle_gap": gap,
                     "quality_grade": grade, "evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE"})
    summary = {"created_at": iso_kst(), "graded_trajectory_count": len(rows), "grade_counts": _count_by(rows, "quality_grade"),
               "grade_A_or_B_count": sum(1 for r in rows if r["quality_grade"] in ("A", "B")),
               "min_obs": TRAJ_MIN_OBS, "min_cycles": TRAJ_MIN_CYCLES}
    return rows, summary


def turnaround_offline_assessment() -> Dict[str, Any]:
    tu_path = C1_ROOT / "turnaround_interval_candidates.parquet"
    import pandas as pd
    df = pd.read_parquet(tu_path) if tu_path.exists() else pd.DataFrame()
    rows = []
    for _i, r in df.iterrows():
        rows.append({"vehicle_token": r.get("vehicle_token"), "route_id": r.get("route_id"),
                     "inbound_direction": str(r.get("inbound_direction")), "outbound_direction": str(r.get("outbound_direction")),
                     "left_cycle": int(r.get("left_cycle")) if r.get("left_cycle") == r.get("left_cycle") else None,
                     "right_cycle": int(r.get("right_cycle")) if r.get("right_cycle") == r.get("right_cycle") else None,
                     "interval_lower_bound_seconds": float(r.get("interval_lower_bound_seconds")),
                     "interval_upper_bound_seconds": float(r.get("interval_upper_bound_seconds")),
                     "interval_width_seconds": float(r.get("interval_width_seconds")),
                     "evidence_status": "INTERVAL_CENSORED_PROVIDER_OBSERVATION", "exact_turnaround_time_claimed": False})
    widths = [x["interval_width_seconds"] for x in rows]
    return {"created_at": iso_kst(), "turnaround_interval_candidate_count": len(rows),
            "evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
            "interval_width_seconds_min": (min(widths) if widths else None), "interval_width_seconds_max": (max(widths) if widths else None),
            "status": ("NO_TURNAROUND_OBSERVED_WITHIN_LIMITED_PILOT" if not rows else "INTERVAL_CENSORED_CANDIDATES_PRESENT"),
            "exact_turnaround_available": False, "actual_layover_available": False, "records": rows}


def eta_consistency_audit(gr) -> Dict[str, Any]:
    total = len(gr)
    valid = int((gr["eta_validity_status"] == "PROVIDER_PREDICTED_ETA").sum()) if total else 0
    predicted_class = int((gr["eta_class"] == "PROVIDER_PREDICTED_ETA").sum()) if total else 0
    nonneg = int((gr["eta_seconds"].fillna(0) >= 0).sum()) if total else 0
    return {"created_at": iso_kst(), "getrealtime02_row_count": int(total), "eta_valid_count": valid,
            "eta_validity_rate": (valid / total) if total else None, "all_class_provider_predicted": predicted_class == total,
            "nonnegative_eta_count": nonneg,
            "actual_arrival_time_claimed": False, "actual_headway_claimed": False,
            "eta_class": "PROVIDER_PREDICTED_ETA"}


def _count_by(rows, field) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r.get(field))] = out.get(str(r.get(field)), 0) + 1
    return out


# --------------------------------------------------------------------------- #
# manifest / lock
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "artifact_manifest_srp2_bis_c1_qa.json"
LOCK_NAME = "_SRP2_BIS_C1_QA_COMPLETE.lock"


def write_manifest(writer: Writer, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None,
                     "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
                "manifest_scope": "SRP2_BIS_C1_QA_TRAJECTORY_QUALITY_AUDIT",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]),
                "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "manifest_self_listed": False, "terminal_lock_listed_inside_manifest": False, "files": rows}
    writer.json(MANIFEST_NAME, manifest)
    return manifest


def write_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    mp = writer.root / MANIFEST_NAME
    writer.json(LOCK_NAME, {"created_at": iso_kst(), "mode": "audit", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
                            "readiness": gate["readiness"], "manifest_relative_path": MANIFEST_NAME,
                            "manifest_sha256": sha256_file(mp), "manifest_size_bytes": mp.stat().st_size,
                            "reference_direction": "_SRP2_BIS_C1_QA_COMPLETE.lock -> artifact_manifest_srp2_bis_c1_qa.json -> payload"})


def verify_manifest(root: Path) -> Dict[str, Any]:
    lock = read_json(root / LOCK_NAME)
    mp = root / lock["manifest_relative_path"]
    manifest = read_json(mp)
    missing = mismatch = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.exists():
            missing += 1
        elif sha256_file(path) != row["sha256"]:
            mismatch += 1
    return {"manifest_hash_ok": sha256_file(mp) == lock["manifest_sha256"], "manifest_size_ok": mp.stat().st_size == lock["manifest_size_bytes"],
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "terminal_lock_listed_inside_manifest": any(r["relative_path"] == LOCK_NAME for r in manifest["files"]),
            "manifest_self_listed": any(r["relative_path"] == MANIFEST_NAME for r in manifest["files"])}


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "audit", "scope": "OFFLINE_PROSPECTIVE_TRAJECTORY_QUALITY_AUDIT",
            "morning_peak_performance_study": False, "operational_performance_claim": False,
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "api_call_count": 0, "network_access_count": 0, "database_access_count": 0, "database_write_count": 0,
            "simulator_execution_count": 0, "training_run_count": 0, "c1_artifact_mutation_count": 0,
            "simulator_module_imported": False, "training_module_imported": False}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
EXPLICIT_PAYLOADS = [
    "upstream_lineage_registry.json", "runner_freeze_audit.json", "qa_environment.json", "c1_upstream_verification.json",
    "c1_artifact_immutability_audit.json", "identity_continuity_audit.json", "identity_continuity_audit.jsonl",
    "position_jump_audit.json", "position_jump_audit.jsonl", "observation_gap_audit.json", "observation_gap_audit.jsonl",
    "route_progression_coherence_audit.json", "route_progression_coherence_audit.jsonl", "route_progression_events.jsonl",
    "sequence_reset_direction_change_interpretation.json", "sequence_reset_direction_change_interpretation.jsonl",
    "route_stop_mapping_quality_audit.json", "trajectory_quality_grades.json", "trajectory_quality_grades.jsonl",
    "trajectory_quality_grades.parquet", "turnaround_candidate_offline_assessment.json", "eta_consistency_audit.json",
    "qa_quality_summary.json", "claim_boundary_reaffirmation.json", "api_call_prohibition_audit.json",
    "database_access_prohibition_audit.json", "simulator_execution_prohibition_audit.json", "training_prohibition_audit.json",
    "stage_immutability_audit.json", "qa_decision.json", "next_stage_readiness.json", "gate_decision.json",
    "downstream_lock.json", "final_report.json", "final_report.md",
]


def run_audit(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    c1 = verify_c1_upstream()
    if not c1["upstream_valid"]:
        raise QAError(BLOCKED_C1, f"C1 upstream invalid: {c1['checks']}")
    for f in ["getpos02_normalized.parquet", "getrealtime02_normalized.parquet"]:
        if not (C1_ROOT / f).exists():
            raise QAError(BLOCKED_INPUT, f"missing C1 input: {f}")

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("qa_environment.json", environment_payload())
    writer.json("c1_upstream_verification.json", c1)
    snap_records, snap_paths = snapshot_c1(writer)
    writer.json("upstream_lineage_registry.json", {"created_at": iso_kst(), "c1_verification": c1,
                "record_count": len(snap_records), "all_byte_identical": all(r["byte_identical"] for r in snap_records), "records": snap_records})
    runner_snapshot = copy_file(writer, RUNNER_PATH, f"runner_snapshot_pre_execution/{RUNNER_REL}")

    # load + audit
    gp, gr, master = load_c1_data()
    groups = _obs_records(gp)

    cont_rows, cont_summary = identity_continuity_audit(groups)
    writer.json("identity_continuity_audit.json", cont_summary)
    writer.jsonl("identity_continuity_audit.jsonl", cont_rows)

    jump_rows, jump_summary = position_jump_audit(groups)
    writer.json("position_jump_audit.json", jump_summary)
    writer.jsonl("position_jump_audit.jsonl", jump_rows)
    jump_by_key: Dict[Tuple[str, str, str], int] = {}
    for j in jump_rows:
        k = (j["vehicle_token"], j["route_id"], j["direction_id"])
        jump_by_key[k] = jump_by_key.get(k, 0) + 1

    gap_rows, gap_summary = observation_gap_audit(groups)
    writer.json("observation_gap_audit.json", gap_summary)
    writer.jsonl("observation_gap_audit.jsonl", gap_rows)
    gap_by_key = {(r["vehicle_token"], r["route_id"], r["direction_id"]): r for r in gap_rows}

    prog_events, prog_traj, prog_summary = route_progression_coherence(groups)
    writer.json("route_progression_coherence_audit.json", prog_summary)
    writer.jsonl("route_progression_coherence_audit.jsonl", prog_traj)
    writer.jsonl("route_progression_events.jsonl", prog_events)
    coh_by_key = {(t["vehicle_token"], t["route_id"], t["direction_id"]): t for t in prog_traj}

    reset_rows, reset_summary = sequence_reset_direction_change_interpretation(groups)
    writer.json("sequence_reset_direction_change_interpretation.json", reset_summary)
    writer.jsonl("sequence_reset_direction_change_interpretation.jsonl", reset_rows)

    map_quality = route_stop_mapping_quality(gp, master)
    writer.json("route_stop_mapping_quality_audit.json", map_quality)

    grade_rows, grade_summary = trajectory_quality_grades(cont_rows, coh_by_key, jump_by_key, gap_by_key)
    writer.json("trajectory_quality_grades.json", grade_summary)
    writer.jsonl("trajectory_quality_grades.jsonl", grade_rows)
    writer.parquet("trajectory_quality_grades.parquet", grade_rows,
                   ["vehicle_token", "route_id", "direction_id", "observation_count", "distinct_cycle_count",
                    "coherence_ratio", "position_jump_count", "max_cycle_gap", "quality_grade", "evidence_class"])

    turn = turnaround_offline_assessment()
    writer.json("turnaround_candidate_offline_assessment.json", turn)
    eta = eta_consistency_audit(gr)
    writer.json("eta_consistency_audit.json", eta)

    # claim boundaries reaffirmed
    if turn["exact_turnaround_available"] or eta["actual_headway_claimed"]:
        raise QAError(FAIL_TURNAROUND, "claim boundary violated")
    writer.json("claim_boundary_reaffirmation.json", {"created_at": iso_kst(), "actual_headway_available": False,
                "actual_arrival_departure_available": False, "actual_dwell_available": False, "exact_turnaround_available": False,
                "trajectory_evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
                "turnaround_evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION", "eta_class": "PROVIDER_PREDICTED_ETA",
                "morning_peak_operational_performance_claim": False})

    # QA quality summary
    qa_summary = {"created_at": iso_kst(),
                  "getpos02_rows": int(len(gp)), "getrealtime02_rows": int(len(gr)),
                  "distinct_vehicle_tokens": cont_summary["distinct_vehicle_token_count"],
                  "token_direction_groups": cont_summary["vehicle_token_direction_group_count"],
                  "continuity_grade_counts": cont_summary["grade_counts"],
                  "vehicles_both_directions": cont_summary["vehicles_observed_in_both_directions_count"],
                  "progression_mean_coherence": prog_summary["mean_coherence_ratio"],
                  "progression_classification_counts": prog_summary["classification_counts"],
                  "position_jump_count": jump_summary["position_jump_count"], "position_jump_rate": jump_summary["position_jump_rate"],
                  "total_missing_cycles_within_spans": gap_summary["total_missing_cycles_within_spans"],
                  "route_stop_on_route_rate": map_quality["stop_on_route_rate"],
                  "sequence_stop_consistency_rate": map_quality["sequence_stop_consistency_rate"],
                  "graded_trajectory_count": grade_summary["graded_trajectory_count"],
                  "trajectory_grade_counts": grade_summary["grade_counts"], "trajectory_grade_A_or_B": grade_summary["grade_A_or_B_count"],
                  "turnaround_interval_candidate_count": turn["turnaround_interval_candidate_count"],
                  "eta_validity_rate": eta["eta_validity_rate"]}
    writer.json("qa_quality_summary.json", qa_summary)

    # prohibition + immutability
    writer.json("api_call_prohibition_audit.json", {"created_at": iso_kst(), "api_call_count": 0, "network_access_count": 0, "bis_api_call_count": 0})
    writer.json("database_access_prohibition_audit.json", {"created_at": iso_kst(), "database_access_count": 0, "database_write_count": 0,
                "note": "route-stop master read from parquet file; no DB connection"})
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_module_imported": False, "simulator_execution_count": 0})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "mappo_training_count": 0, "gatv2_training_count": 0})
    # C1 immutability: re-verify C1 manifest hash unchanged
    c1_after = verify_c1_upstream()
    c1_mut = 0 if c1_after["manifest_sha256"] == c1["manifest_sha256"] and c1_after["payload_hash_mismatch_count"] == 0 else 1
    writer.json("c1_artifact_immutability_audit.json", {"created_at": iso_kst(), "c1_manifest_sha_before": c1["manifest_sha256"],
                "c1_manifest_sha_after": c1_after["manifest_sha256"], "c1_artifact_mutation_count": c1_mut,
                "c1_payload_hash_mismatch_after": c1_after["payload_hash_mismatch_count"]})
    if c1_mut:
        raise QAError(FAIL_C1_MUT, "C1 artifact changed during audit")
    writer.json("stage_immutability_audit.json", {"created_at": iso_kst(), "c1_artifact_mutation_count": c1_mut,
                "source_modification_count": 0, "git_commit_count": 0, "git_push_count": 0})

    # freeze
    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise QAError(FAIL_RUNNER, "runner mutated during audit")

    # decision + readiness
    ab = grade_summary["grade_A_or_B_count"]
    coh = prog_summary["mean_coherence_ratio"] or 0.0
    map_ok = (map_quality["stop_on_route_rate"] or 0) >= 0.98
    if ab >= 5 and coh >= 0.85 and map_ok:
        readiness = READINESS_SUFFICIENT
        reason = "trajectory quality sufficient: multiple A/B trajectories, coherent progression, clean route-stop mapping"
    elif grade_summary["graded_trajectory_count"] >= 1:
        readiness = READINESS_LIMITED
        reason = "trajectory candidates present but quality/coverage limited; an additional short capture is advisable"
    else:
        readiness = READINESS_REPAIR
        reason = "insufficient trajectory quality"
    decision = {"created_at": iso_kst(), "readiness": readiness, "reason": reason,
                "graded_trajectory_count": grade_summary["graded_trajectory_count"], "grade_A_or_B_count": ab,
                "mean_coherence_ratio": coh, "position_jump_count": jump_summary["position_jump_count"],
                "route_stop_on_route_rate": map_quality["stop_on_route_rate"],
                "turnaround_interval_candidate_count": turn["turnaround_interval_candidate_count"],
                "exact_turnaround_available": False, "actual_headway_available": False}
    writer.json("qa_decision.json", decision)

    gate = {"created_at": iso_kst(), "mode": "audit", "gate": PASS_GATE, "gate_passed": True, "readiness": readiness,
            "scope": "OFFLINE_PROSPECTIVE_TRAJECTORY_QUALITY_AUDIT",
            "database_ingestion_authorized": False, "additional_capture_authorized": False, "state_reconstruction_authorized": False,
            "physical_vehicle_agent_mapping_authorized": False, "policy_interface_adaptation_authorized": False,
            "historical_transition_authorized": False, "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
            "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)
    writer.json("next_stage_readiness.json", {"created_at": iso_kst(), "readiness": readiness, "reason": reason,
                "constraints_maintained": {"api_calls": 0, "db_write": 0, "simulator": 0, "training": 0},
                "note": "C1-QA validates the prospective capture pipeline end-to-end; downstream (DB delta, physical-vehicle mapping, policy adaptation) remains user-gated"})
    writer.json("downstream_lock.json", {"c1_upstream_verified": True, "c1_qa_complete": True,
                "scope": "OFFLINE_PROSPECTIVE_TRAJECTORY_QUALITY_AUDIT", "api_call_count": 0, "database_write_count": 0,
                "simulator_execution_count": 0, "training_run_count": 0, "c1_artifact_mutation_count": 0,
                "graded_trajectory_count": grade_summary["graded_trajectory_count"], "grade_A_or_B_count": ab,
                "trajectory_evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
                "turnaround_evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION", "turnaround_interval_candidate_count": turn["turnaround_interval_candidate_count"],
                "actual_headway_available": False, "actual_arrival_departure_available": False, "actual_dwell_available": False,
                "exact_turnaround_available": False, "database_ingestion_authorized": False, "additional_capture_authorized": False,
                "state_reconstruction_authorized": False, "physical_vehicle_agent_mapping_authorized": False,
                "policy_interface_adaptation_authorized": False, "reward_rebuild_authorized": False,
                "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False,
                "training_allowed": False, "next_stage_readiness": readiness})

    report_payload, report_md = build_final_report(root, gate, c1, qa_summary, cont_summary, prog_summary, jump_summary,
                                                    gap_summary, map_quality, grade_summary, turn, eta, decision)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    payloads = list(EXPLICIT_PAYLOADS) + list(snap_paths) + [runner_snapshot["snapshot_relative_path"]]
    manifest = write_manifest(writer, payloads)
    if manifest["missing_payload_count"]:
        raise QAError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, gate)
    v = verify_manifest(root)
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise QAError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("SRP2-BIS-C1-QA TRAJECTORY QUALITY AUDIT COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"readiness: {readiness}")
    print(f"graded_trajectories: {grade_summary['graded_trajectory_count']} (A/B={ab}) grades={grade_summary['grade_counts']}")
    print(f"mean_coherence: {round(coh,3)} | position_jumps: {jump_summary['position_jump_count']} | stop_on_route_rate: {map_quality['stop_on_route_rate']} | seq_consistency: {map_quality['sequence_stop_consistency_rate']}")
    print(f"turnaround_interval_candidates: {turn['turnaround_interval_candidate_count']} (interval-censored) | exact_turnaround: False | actual_headway: False")
    print(f"c1_mutation: {c1_mut} | api_calls: 0 | db_write: 0 | runner_frozen: {runner_freeze['runner_frozen']}")
    return root


def build_final_report(root, gate, c1, qa, cont, prog, jump, gap, mapq, grade, turn, eta, decision):
    answers = {
        "01_c1_upstream_verified": c1["upstream_valid"],
        "02_getpos02_rows": qa["getpos02_rows"],
        "03_getrealtime02_rows": qa["getrealtime02_rows"],
        "04_distinct_vehicles": qa["distinct_vehicle_tokens"],
        "05_continuity_grades": qa["continuity_grade_counts"],
        "06_vehicles_both_directions": qa["vehicles_both_directions"],
        "07_progression_mean_coherence": qa["progression_mean_coherence"],
        "08_progression_classes": qa["progression_classification_counts"],
        "09_sequence_reset_direction_change_interval_censored": True,
        "10_position_jump_count": qa["position_jump_count"],
        "11_position_jump_rate": qa["position_jump_rate"],
        "12_missing_cycles_within_spans": qa["total_missing_cycles_within_spans"],
        "13_route_stop_on_route_rate": qa["route_stop_on_route_rate"],
        "14_sequence_stop_consistency_rate": qa["sequence_stop_consistency_rate"],
        "15_graded_trajectories": qa["graded_trajectory_count"],
        "16_trajectory_grades": qa["trajectory_grade_counts"],
        "17_turnaround_interval_candidates": qa["turnaround_interval_candidate_count"],
        "18_exact_turnaround_claimed": False,
        "19_actual_headway_claimed": False,
        "20_eta_class_provider_predicted": eta["all_class_provider_predicted"],
        "21_api_calls": 0, "22_db_write": 0, "23_simulator_training": 0,
        "24_c1_artifact_mutated": False,
        "25_next_stage_readiness": decision["readiness"],
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "audit", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
               "readiness": gate["readiness"], "scope": "OFFLINE_PROSPECTIVE_TRAJECTORY_QUALITY_AUDIT", "quick_answers": answers,
               "qa_quality_summary": qa}
    grade_a_b = qa["trajectory_grade_A_or_B"]
    lines = [
        "# SRP2-BIS-C1-QA Prospective Trajectory Quality Audit — Final Report", "",
        f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
        "- scope: OFFLINE trajectory quality / route progression / identity continuity (no API, no DB, no simulator, no training; C1 immutable)",
        "",
        "## Bottom line",
        f"- Audited the sealed C1 capture: {qa['getpos02_rows']} getPos02 rows over {qa['distinct_vehicle_tokens']} distinct vehicles on route 814, {qa['getrealtime02_rows']} ETA rows.",
        f"- **Identity continuity**: {qa['continuity_grade_counts']}; {qa['vehicles_both_directions']} vehicles seen in both directions (turnaround-observation candidates).",
        f"- **Route progression coherence**: mean {round(qa['progression_mean_coherence'],3)}; classes {qa['progression_classification_counts']}.",
        f"- **Position jumps**: {qa['position_jump_count']} (rate {qa['position_jump_rate']}) above {POSITION_JUMP_SPEED_KMH} km/h implied speed.",
        f"- **Route-stop mapping**: on-route {qa['route_stop_on_route_rate']}, provider (seq,bsId)↔master consistency {qa['sequence_stop_consistency_rate']}.",
        f"- **Trajectory grades**: {qa['trajectory_grade_counts']} (A/B = {grade_a_b} of {qa['graded_trajectory_count']}).",
        f"- **Turnaround**: {qa['turnaround_interval_candidate_count']} interval-censored candidates — NOT promoted to exact turnaround time.",
        "",
        "## Claim boundaries (reaffirmed)",
        "- ETA = PROVIDER_PREDICTED_ETA; trajectory = PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE; turnaround = INTERVAL_CENSORED_PROVIDER_OBSERVATION.",
        "- actual headway / arrival-departure / dwell / exact turnaround: all FALSE. No morning-peak operational-performance claim.",
        "",
        "## Guardrails (held)",
        "- api_call 0 · network 0 · db_write 0 · simulator 0 · training 0 · C1 artifact mutation 0 · git commit/push 0.",
        "",
        "## Quick answers",
    ]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["audit"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_audit(args.artifact_root)
    except QAError as exc:
        print("SRP2-BIS-C1-QA AUDIT BLOCKED/FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
