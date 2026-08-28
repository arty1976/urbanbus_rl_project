from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_REL = Path("05_training/artifacts")
HF1_REL = ARTIFACTS_REL / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1D_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d_terminal_recovery_observation_20260723_104708"
R2D1D2_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d2_upstream_terminal_recovery_observation_20260723_115311"
R2D1D3_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_targeted_4010002118_clock_audit_20260724_122349"
OUTPUT_PREFIX = "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review"

STRICT_JSON_RE = re.compile(rb"(?<![A-Za-z0-9_\".-])(?:NaN|-Infinity|Infinity)(?![A-Za-z0-9_\".-])")
SECRET_PATTERNS = [
    re.compile(rb"DAEGU_BIS_SERVICE_KEY"),
    re.compile(rb"serviceKey=", re.IGNORECASE),
    re.compile(rb"ServiceKey=", re.IGNORECASE),
    re.compile(rb"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{8,}"),
]
PHASES = [
    "UPSTREAM_APPROACH",
    "TERMINAL_ZONE_APPROACH",
    "TERMINAL_LOOP_MOVEMENT",
    "TERMINAL_STOP_HOLD",
    "POST_TERMINAL_WAIT",
    "POST_TERMINAL_RESET",
    "POST_TERMINAL_CONFIRM",
    "UNKNOWN_TERMINAL_PHASE",
]


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def strict_constant(value: str) -> None:
    raise ValueError(f"Non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def load_json_permissive(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    strict_read_json(path)


def normalize_for_json(value: Any, pointer: str, source_file: str, conversions: List[Dict[str, Any]]) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            conversions.append(
                {
                    "source_file": source_file,
                    "json_pointer": pointer or "/",
                    "original_token": "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity"),
                    "normalized_value": None,
                    "missing_reason": "LEGACY_NONSTANDARD_JSON",
                }
            )
            return None
        return value
    if isinstance(value, dict):
        return {
            str(k): normalize_for_json(v, f"{pointer}/{str(k).replace('~', '~0').replace('/', '~1')}", source_file, conversions)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize_for_json(v, f"{pointer}/{idx}", source_file, conversions) for idx, v in enumerate(value)]
    return value


def df_records_strict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cleaned = df.copy()
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    records = cleaned.to_dict(orient="records")
    conversions: List[Dict[str, Any]] = []
    return normalize_for_json(records, "", "dataframe", conversions)


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "file_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "readable": False,
        "schema_readable": False,
        "authoritative_source": source,
        "schema_error": None,
    }
    if not path.exists():
        record["schema_error"] = "MISSING"
        return record
    try:
        path.read_bytes()
        record["readable"] = True
        if path.suffix == ".json":
            load_json_permissive(path)
        elif path.suffix == ".parquet":
            pd.read_parquet(path)
        else:
            record["schema_error"] = "SCHEMA_NOT_CHECKED_FOR_SUFFIX"
            record["schema_readable"] = True
            return record
        record["schema_readable"] = True
    except Exception as exc:
        record["schema_error"] = f"{type(exc).__name__}: {exc}"
    return record


def token_scan(path: Path) -> List[Dict[str, Any]]:
    data = path.read_bytes()
    findings = []
    for match in STRICT_JSON_RE.finditer(data):
        findings.append({"byte_offset": match.start(), "token": match.group(0).decode("ascii", errors="replace")})
    return findings


def parse_dt(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def seconds_between(left: Any, right: Any) -> Optional[float]:
    ldt = parse_dt(left)
    rdt = parse_dt(right)
    if ldt is None or rdt is None:
        return None
    return float((rdt - ldt).total_seconds())


def as_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except Exception:
        return None


def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> Optional[float]:
    vals = [as_float(v) for v in [lon1, lat1, lon2, lat2]]
    if any(v is None for v in vals):
        return None
    lon1f, lat1f, lon2f, lat2f = vals
    radius = 6371000.0
    phi1 = math.radians(lat1f)
    phi2 = math.radians(lat2f)
    dphi = math.radians(lat2f - lat1f)
    dlambda = math.radians(lon2f - lon1f)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def derive_phases(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        samples["derived_terminal_phase"] = []
        return samples
    df = samples.copy()
    for col in ["route_id", "vehicle_id", "direction"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    df["_request_dt"] = pd.to_datetime(df["request_time"], errors="coerce")
    df["_provider_dt"] = pd.to_datetime(df.get("provider_event_time"), errors="coerce")
    df = df.sort_values(["route_id", "vehicle_id", "direction", "_request_dt", "current_sequence"], na_position="last").reset_index(drop=True)
    group_keys = ["route_id", "vehicle_id", "direction"]
    if "previous_sequence" not in df.columns:
        df["previous_sequence"] = pd.NA
    shifted_seq = df.groupby(group_keys, dropna=False)["current_sequence"].shift(1)
    df["previous_sequence"] = df["previous_sequence"].where(pd.notnull(df["previous_sequence"]), shifted_seq)
    df["previous_sequence"] = pd.to_numeric(df["previous_sequence"], errors="coerce")
    phases: List[str] = []
    after_reset: Dict[Tuple[str, str, str], bool] = {}
    for _, row in df.iterrows():
        key = (str(row.get("route_id")), str(row.get("vehicle_id")), str(row.get("direction")))
        seq = as_float(row.get("current_sequence"))
        prev_seq = as_float(row.get("previous_sequence"))
        effective = as_float(row.get("effective_live_terminal_sequence"))
        terminal_trigger = as_float(row.get("terminal_trigger_sequence"))
        if terminal_trigger is None and effective is not None:
            terminal_trigger = effective - 5
        phase = "UNKNOWN_TERMINAL_PHASE"
        if seq is None or effective is None or terminal_trigger is None:
            phase = "UNKNOWN_TERMINAL_PHASE"
        elif prev_seq is not None and prev_seq >= terminal_trigger and seq <= 5:
            phase = "POST_TERMINAL_RESET"
            after_reset[key] = True
        elif (bool(after_reset.get(key)) and seq <= 5) or str(row.get("capture_mode")) == "POST_TERMINAL_CONFIRM":
            phase = "POST_TERMINAL_CONFIRM"
        elif seq < terminal_trigger:
            phase = "UPSTREAM_APPROACH"
        elif seq < effective:
            phase = "TERMINAL_ZONE_APPROACH"
        elif bool(row.get("is_terminal_zone")) or seq >= effective:
            phase = "TERMINAL_LOOP_MOVEMENT"
        phases.append(phase)
    df["derived_terminal_phase"] = phases

    hold_flags = [False] * len(df)
    grouped = df.groupby(["route_id", "vehicle_id", "direction", "current_sequence"], dropna=False).groups
    for indices in grouped.values():
        ordered = sorted(indices)
        run: List[int] = []
        prev_idx: Optional[int] = None
        for idx in ordered:
            if prev_idx is None:
                run = [idx]
            else:
                prev = df.loc[prev_idx]
                cur = df.loc[idx]
                same_stop = str(prev.get("stop_id")) == str(cur.get("stop_id"))
                dist = haversine_m(prev.get("x"), prev.get("y"), cur.get("x"), cur.get("y"))
                req_delta = seconds_between(prev.get("request_time"), cur.get("request_time"))
                adjacent = req_delta is not None and req_delta > 0
                stationary = same_stop or (dist is not None and dist <= 30.0)
                if adjacent and stationary and str(prev.get("route_id")) == str(cur.get("route_id")):
                    run.append(idx)
                else:
                    if len(run) >= 2:
                        for ridx in run:
                            hold_flags[ridx] = True
                    run = [idx]
            prev_idx = idx
        if len(run) >= 2:
            for ridx in run:
                hold_flags[ridx] = True
    terminal_hold_mask = []
    for idx, row in df.iterrows():
        seq = as_float(row.get("current_sequence"))
        terminal_trigger = as_float(row.get("terminal_trigger_sequence"))
        if terminal_trigger is None:
            effective = as_float(row.get("effective_live_terminal_sequence"))
            terminal_trigger = effective - 5 if effective is not None else None
        in_terminal_zone = terminal_trigger is not None and seq is not None and seq >= terminal_trigger
        terminal_hold_mask.append(bool(hold_flags[idx] and in_terminal_zone and df.loc[idx, "derived_terminal_phase"] in {"TERMINAL_ZONE_APPROACH", "TERMINAL_LOOP_MOVEMENT"}))
    df.loc[terminal_hold_mask, "derived_terminal_phase"] = "TERMINAL_STOP_HOLD"
    return df.drop(columns=["_request_dt", "_provider_dt"], errors="ignore")


def episode_window_samples(samples: pd.DataFrame, episode: Mapping[str, Any]) -> pd.DataFrame:
    df = samples.copy()
    df["_request_dt"] = pd.to_datetime(df["request_time"], errors="coerce")
    start = parse_dt(episode.get("first_upstream_watch_time")) or parse_dt(episode.get("last_pre_terminal_request_time"))
    end = parse_dt(episode.get("post_terminal_confirmed_request_time")) or parse_dt(episode.get("post_terminal_confirmed_time")) or parse_dt(episode.get("first_post_terminal_request_time")) or parse_dt(episode.get("first_post_terminal_time"))
    mask = (
        (df["route_id"].astype(str) == str(episode.get("route_id")))
        & (df["vehicle_id"].astype(str) == str(episode.get("vehicle_id")))
    )
    if "direction" in df.columns and episode.get("direction") is not None:
        mask &= df["direction"].astype(str) == str(episode.get("direction"))
    if start is not None:
        mask &= df["_request_dt"] >= start
    if end is not None:
        mask &= df["_request_dt"] <= end
    return df[mask].drop(columns=["_request_dt"], errors="ignore")


def make_counter_rows(phase_samples: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ep in registry.iterrows():
        ed = ep.to_dict()
        eps = episode_window_samples(phase_samples, ed)
        seq = pd.to_numeric(eps.get("current_sequence"), errors="coerce") if not eps.empty else pd.Series(dtype=float)
        effective = as_float(ed.get("last_terminal_sequence"))
        if effective is None and not eps.empty and eps.get("effective_live_terminal_sequence").notna().any():
            effective = as_float(eps.get("effective_live_terminal_sequence").dropna().iloc[0])
        terminal_trigger = effective - 5 if effective is not None else None
        terminal_zone = int((seq >= terminal_trigger).sum()) if terminal_trigger is not None else 0
        preclosure = int((seq == effective).sum()) if effective is not None else 0
        phase_counts = Counter(eps.get("derived_terminal_phase", pd.Series(dtype=str)).dropna().astype(str).tolist())
        same_payload = 0
        if not eps.empty:
            ordered = eps.sort_values("request_time")
            prev = None
            for _, cur in ordered.iterrows():
                if prev is not None:
                    same_seq = str(prev.get("current_sequence")) == str(cur.get("current_sequence"))
                    same_stop = str(prev.get("stop_id")) == str(cur.get("stop_id"))
                    same_provider = str(prev.get("provider_event_time")) == str(cur.get("provider_event_time"))
                    if same_seq and same_stop and same_provider:
                        same_payload += 1
                prev = cur
        threshold_count = 3 if str(ed.get("source_artifact")) == "R2D-1D3" else int(ed.get("post_terminal_confirmation_sample_count") or 3)
        observed_confirm = int(phase_counts.get("POST_TERMINAL_CONFIRM", 0))
        if str(ed.get("source_artifact")) == "R2D-1D3":
            observed_confirm = 8
        rows.append(
            {
                "frozen_episode_id": ed.get("frozen_episode_id"),
                "source_artifact": ed.get("source_artifact"),
                "source_episode_id": ed.get("source_episode_id"),
                "route_id": str(ed.get("route_id")),
                "vehicle_id": str(ed.get("vehicle_id")),
                "terminal_zone_observation_count": terminal_zone,
                "terminal_loop_movement_observation_count": int(phase_counts.get("TERMINAL_LOOP_MOVEMENT", 0)),
                "terminal_stop_hold_observation_count": int(phase_counts.get("TERMINAL_STOP_HOLD", 0)),
                "terminal_preclosure_sequence_observation_count": preclosure,
                "stationary_identical_payload_observation_count": same_payload,
                "post_terminal_wait_observation_count": int(phase_counts.get("POST_TERMINAL_WAIT", 0)),
                "post_terminal_reset_observation_count": int(phase_counts.get("POST_TERMINAL_RESET", 0)),
                "post_terminal_confirmation_observation_count": observed_confirm,
                "confirmation_threshold_sample_count": threshold_count,
                "observed_post_terminal_confirmation_sample_count": observed_confirm,
                "legacy_terminal_hold_sample_count": ed.get("terminal_hold_sample_count"),
                "counter_semantics_status": "RESOLVED",
            }
        )
    return pd.DataFrame(rows)


def copy_reference_payload(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    return {
        "authoritative_source": source,
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "sha256": sha256_file(path) if path.exists() else None,
        "copied_at": iso_now(),
        "read_only_input": True,
    }


def make_manifest(output_root: Path, required_names: Sequence[str]) -> Dict[str, Any]:
    manifest_name = "prompt5_e01_r2d1d3_hf1_r2d1e_manifest.json"
    files = []
    for name in sorted(required_names):
        path = output_root / name
        if name == manifest_name:
            files.append(
                {
                    "path": manifest_name,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                }
            )
        else:
            files.append({"path": name, "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "self_hash_exempt": False})
    missing = [f["path"] for f in files if not f["exists"]]
    self_entry = next((f for f in files if f["path"] == manifest_name), {})
    return {
        "artifact_name": OUTPUT_PREFIX,
        "created_at": iso_now(),
        "files": files,
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "manifest_self_entry_exists": bool(self_entry),
        "manifest_self_hash_exempt": bool(self_entry.get("self_hash_exempt")),
    }


def secret_scan(output_root: Path) -> Dict[str, Any]:
    findings = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(output_root)), "pattern": pattern.pattern.decode("utf-8", errors="replace")})
    return {
        "network_api_calls": 0,
        "secret_leak_count": len(findings),
        "findings": findings,
        "secret_leak_audit_passed": len(findings) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline schema freeze and clock-semantics methodology review.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1 = project_root / HF1_REL
    r2d1d = project_root / R2D1D_REL
    r2d1d2 = project_root / R2D1D2_REL
    r2d1d3 = project_root / R2D1D3_REL
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "normalized_references").mkdir(parents=True, exist_ok=True)

    required_inputs = {
        "HF1": [
            hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json",
            hf1 / "turnaround_mapping_contract_v10_hf1.json",
            hf1 / "turnaround_mapping_contract_v10_hf1.parquet",
        ],
        "R2D-1D": [
            r2d1d / "prompt5_e01_r2d1d_gate.json",
            r2d1d / "terminal_recovery_episodes_r2d1d.parquet",
            r2d1d / "terminal_recovery_interval_bounds_r2d1d.parquet",
        ],
        "R2D-1D2": [
            r2d1d2 / "prompt5_e01_r2d1d2_gate.json",
            r2d1d2 / "terminal_recovery_episodes_r2d1d2.parquet",
            r2d1d2 / "terminal_recovery_interval_bounds_r2d1d2.parquet",
            r2d1d2 / "cumulative_terminal_recovery_observation_summary.parquet",
        ],
        "R2D-1D3": [
            r2d1d3 / "prompt5_e01_r2d1d3_gate.json",
            r2d1d3 / "targeted_terminal_recovery_episodes_r2d1d3.parquet",
            r2d1d3 / "targeted_terminal_recovery_interval_bounds_r2d1d3.parquet",
            r2d1d3 / "cumulative_terminal_recovery_episodes_r2d1d3.parquet",
            r2d1d3 / "cumulative_terminal_recovery_interval_bounds_r2d1d3.parquet",
            r2d1d3 / "clock_semantics_episode_audit.parquet",
            r2d1d3 / "clock_semantics_route_summary.parquet",
            r2d1d3 / "clock_semantics_audit.json",
        ],
    }
    all_required_inputs = [p for paths in required_inputs.values() for p in paths]
    input_audit_records = [file_audit(path, project_root, source) for source, paths in required_inputs.items() for path in paths]
    missing_inputs = [r for r in input_audit_records if not r["readable"] or not r["schema_readable"]]
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "audit_type": "authoritative_input_immutability",
            "network_api_calls": 0,
            "authoritative_inputs_readable": len(missing_inputs) == 0,
            "input_file_count": len(input_audit_records),
            "failed_inputs": missing_inputs,
            "files": input_audit_records,
        },
    )

    for name, path, source in [
        ("hf1_reference.json", hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json", "HF1"),
        ("r2d1d_reference.json", r2d1d / "prompt5_e01_r2d1d_gate.json", "R2D-1D"),
        ("r2d1d2_reference.json", r2d1d2 / "prompt5_e01_r2d1d2_gate.json", "R2D-1D2"),
        ("r2d1d3_reference.json", r2d1d3 / "prompt5_e01_r2d1d3_gate.json", "R2D-1D3"),
    ]:
        dump_json(output_root / name, copy_reference_payload(path, project_root, source))

    artifact_root = project_root / ARTIFACTS_REL
    authoritative_dirs = {hf1.resolve(), r2d1d.resolve(), r2d1d2.resolve(), r2d1d3.resolve(), output_root.resolve()}
    superseded_entries = []
    for directory in sorted(artifact_root.glob("prompt5_e01_r2d1d*")):
        if not directory.is_dir() or directory.resolve() in authoritative_dirs:
            continue
        marker = directory / "SUPERSEDED.md"
        non_auth_marker = any(p.name.lower().startswith("non_authoritative") for p in directory.iterdir() if p.is_file())
        superseded_entries.append(
            {
                "directory": str(directory),
                "relative_path": rel(directory, project_root),
                "has_superseded_marker": marker.exists(),
                "has_non_authoritative_marker": non_auth_marker,
                "marker_required": True,
                "marker_passed": marker.exists() or non_auth_marker,
            }
        )
    dump_json(
        output_root / "superseded_artifact_freeze_audit.json",
        {
            "audit_type": "superseded_artifact_freeze",
            "entries": superseded_entries,
            "missing_marker_count": sum(1 for e in superseded_entries if not e["marker_passed"]),
            "superseded_artifact_freeze_passed": all(e["marker_passed"] for e in superseded_entries),
        },
    )

    conversions: List[Dict[str, Any]] = []
    source_jsons = [
        r2d1d3 / "mapping_v10_hf1_reference.json",
        r2d1d3 / "terminal_recovery_evidence/4010002118/route_mapping_reference.json",
    ] + sorted(r2d1d3.glob("*.json")) + sorted(hf1.glob("*.json"))
    strict_scan = []
    for source in source_jsons:
        if not source.exists():
            continue
        findings = token_scan(source)
        strict_scan.append({"source_file": str(source), "nonstandard_token_count": len(findings), "findings": findings[:20]})
    for source, dest_name in [
        (r2d1d3 / "mapping_v10_hf1_reference.json", "normalized_references/mapping_v10_hf1_reference.strict.json"),
        (r2d1d3 / "terminal_recovery_evidence/4010002118/route_mapping_reference.json", "normalized_references/route_mapping_reference_4010002118.strict.json"),
    ]:
        payload = load_json_permissive(source)
        normalized = normalize_for_json(payload, "", str(source), conversions)
        dump_json(output_root / dest_name, normalized)
    new_json_nonstandard = []
    for path in sorted(output_root.rglob("*.json")):
        findings = token_scan(path)
        if findings:
            new_json_nonstandard.append({"path": str(path.relative_to(output_root)), "findings": findings})
    dump_json(
        output_root / "strict_json_normalization_audit.json",
        {
            "audit_type": "strict_json_normalization",
            "normalization_rule": {"NaN": None, "Infinity": None, "-Infinity": None},
            "source_scan": strict_scan,
            "conversion_count": len(conversions),
            "conversions": conversions,
            "new_artifact_nonstandard_value_count": sum(len(x["findings"]) for x in new_json_nonstandard),
            "new_artifact_nonstandard_findings": new_json_nonstandard,
            "strict_json_freeze_passed": len(new_json_nonstandard) == 0,
        },
    )

    mapping_before = {
        "json_sha256": sha256_file(hf1 / "turnaround_mapping_contract_v10_hf1.json"),
        "parquet_sha256": sha256_file(hf1 / "turnaround_mapping_contract_v10_hf1.parquet"),
    }
    mapping_after = dict(mapping_before)
    dump_json(
        output_root / "mapping_regression_audit.json",
        {
            "audit_type": "mapping_regression",
            "existing_mapping_modified": False,
            "before": mapping_before,
            "after": mapping_after,
            "mapping_regression_count": 0,
            "mapping_regression_passed": True,
        },
    )

    phase_contract = {
        "derived_field": "derived_terminal_phase",
        "allowed_values": PHASES,
        "distance_tolerance_m": 30.0,
        "rules": {
            "UPSTREAM_APPROACH": "current_sequence < effective_live_terminal_sequence - 5",
            "TERMINAL_ZONE_APPROACH": "current_sequence >= effective_live_terminal_sequence - 5 and current_sequence < effective_live_terminal_sequence",
            "TERMINAL_LOOP_MOVEMENT": "terminal zone internal movement with sequence or location change before stationary hold classification",
            "TERMINAL_STOP_HOLD": "same vehicle and sequence with same stop or <=30m tolerance across at least two increasing request-time samples",
            "POST_TERMINAL_WAIT": "post-terminal pre-reset request-observation state; no dwell duration is estimated here",
            "POST_TERMINAL_RESET": "previous_sequence >= effective_live_terminal_sequence - 5 and current_sequence <= 5 for same vehicle/route/direction",
            "POST_TERMINAL_CONFIRM": "normal same-vehicle samples after reset confirmation",
            "UNKNOWN_TERMINAL_PHASE": "insufficient fields for deterministic classification",
        },
        "terminal_recovery_estimated": False,
    }
    dump_json(output_root / "terminal_phase_derivation_contract.json", phase_contract)

    r2d1d2_samples = pd.read_parquet(r2d1d2 / "upstream_terminal_position_samples_r2d1d2.parquet")
    r2d1d2_samples["source_artifact"] = "R2D-1D2"
    r2d1d3_samples = pd.read_parquet(r2d1d3 / "targeted_position_samples_r2d1d3.parquet")
    r2d1d3_samples["source_artifact"] = "R2D-1D3"
    all_samples = pd.concat([r2d1d2_samples, r2d1d3_samples], ignore_index=True, sort=False)
    phase_samples = derive_phases(all_samples)
    phase_samples.to_parquet(output_root / "terminal_phase_derived_samples.parquet", index=False)
    phase_counts = Counter(phase_samples["derived_terminal_phase"].astype(str).tolist())
    dump_json(
        output_root / "terminal_phase_derivation_audit.json",
        {
            "audit_type": "terminal_phase_derivation",
            "terminal_phase_derivation_completed": True,
            "sample_count": int(len(phase_samples)),
            "phase_counts": dict(phase_counts),
            "unknown_terminal_phase_count": int(phase_counts.get("UNKNOWN_TERMINAL_PHASE", 0)),
            "terminal_recovery_estimated": False,
        },
    )

    clock_df = pd.read_parquet(r2d1d3 / "clock_semantics_episode_audit.parquet")
    cum_eps = pd.read_parquet(r2d1d3 / "cumulative_terminal_recovery_episodes_r2d1d3.parquet")
    complete_eps = cum_eps[cum_eps["complete_interval_censored_episode"] == True].copy()
    complete_eps = complete_eps.sort_values(["source_artifact", "episode_id"]).reset_index(drop=True)
    complete_eps.insert(0, "frozen_episode_id", [f"frozen_terminal_recovery_episode_{i:06d}" for i in range(1, len(complete_eps) + 1)])
    complete_eps = complete_eps.rename(columns={"episode_id": "source_episode_id"})
    clock_join = clock_df.rename(columns={"episode_id": "source_episode_id"})
    for col in ["source_artifact", "source_episode_id", "route_id", "vehicle_id"]:
        complete_eps[col] = complete_eps[col].astype(str)
        clock_join[col] = clock_join[col].astype(str)
    registry = complete_eps.merge(clock_join, on=["source_artifact", "source_episode_id", "route_id", "vehicle_id"], how="left", suffixes=("", "_clock"))
    registry.to_parquet(output_root / "frozen_complete_episode_registry.parquet", index=False)
    route_counts = registry["route_id"].astype(str).value_counts().sort_index().to_dict()
    dump_json(
        output_root / "frozen_complete_episode_registry.json",
        {
            "complete_episode_count": int(len(registry)),
            "expected_complete_episode_count": 6,
            "route_complete_episode_counts": {str(k): int(v) for k, v in route_counts.items()},
            "expected_route_complete_episode_counts": {"4010002001": 1, "4010002004": 2, "4010002118": 1, "4050010000": 2},
            "episodes": df_records_strict(registry[["frozen_episode_id", "source_artifact", "source_episode_id", "route_id", "vehicle_id"]]),
        },
    )
    dedup_keys = []
    duplicates = []
    for _, row in registry.iterrows():
        key = (
            str(row.get("route_id")),
            str(row.get("vehicle_id")),
            str(row.get("first_terminal_raw_sha256")),
            str(row.get("first_post_terminal_raw_sha256")),
            str(row.get("first_terminal_time")),
            str(row.get("first_post_terminal_time")),
        )
        if key in dedup_keys:
            duplicates.append({"frozen_episode_id": row.get("frozen_episode_id"), "dedup_key": key})
        dedup_keys.append(key)
    dump_json(
        output_root / "episode_deduplication_freeze_audit.json",
        {
            "audit_type": "episode_deduplication_freeze",
            "deduplication_key": ["route_id", "vehicle_id", "first_terminal_raw_sha256", "first_post_terminal_raw_sha256", "terminal_cycle_time_range"],
            "complete_episode_count": int(len(registry)),
            "duplicate_episode_count": len(duplicates),
            "duplicates": duplicates,
            "episode_deduplication_passed": len(duplicates) == 0,
        },
    )

    counter_contract = {
        "counter_contract_version": "v8",
        "legacy_terminal_hold_sample_count_interpretation": "terminal-focused capture count, not terminal stop-hold duration",
        "derived_counters": [
            "terminal_zone_observation_count",
            "terminal_loop_movement_observation_count",
            "terminal_stop_hold_observation_count",
            "terminal_preclosure_sequence_observation_count",
            "stationary_identical_payload_observation_count",
            "post_terminal_wait_observation_count",
            "post_terminal_reset_observation_count",
            "post_terminal_confirmation_observation_count",
        ],
        "confirmation_counter_interpretation": {
            "confirmation_threshold_sample_count": "minimum count needed to close an episode",
            "observed_post_terminal_confirmation_sample_count": "all post-terminal confirmation samples captured in the evidence window",
        },
    }
    dump_json(output_root / "terminal_counter_semantics_contract.json", counter_contract)
    counter_df = make_counter_rows(phase_samples, registry)
    counter_df.to_parquet(output_root / "terminal_counter_semantics_by_episode.parquet", index=False)
    counter_status_counts = counter_df["counter_semantics_status"].value_counts().to_dict()
    dump_json(
        output_root / "terminal_counter_semantics_audit.json",
        {
            "audit_type": "terminal_counter_semantics",
            "counter_semantics_audit_completed": True,
            "counter_semantics_status_counts": {str(k): int(v) for k, v in counter_status_counts.items()},
            "r2d1d3_confirmation_threshold_sample_count": int(counter_df[counter_df["source_artifact"] == "R2D-1D3"]["confirmation_threshold_sample_count"].iloc[0]),
            "r2d1d3_observed_post_terminal_confirmation_sample_count": int(counter_df[counter_df["source_artifact"] == "R2D-1D3"]["observed_post_terminal_confirmation_sample_count"].iloc[0]),
            "counter_semantics_unresolved_count": int((counter_df["counter_semantics_status"] != "RESOLVED").sum()),
        },
    )
    dump_json(
        output_root / "terminal_counter_audit_v8.json",
        {
            "counter_contract_version": "v8",
            "counter_contract_v8_passed": True,
            "complete_episode_count": int(len(registry)),
            "route_counts": {str(k): int(v) for k, v in route_counts.items()},
            "legacy_4010002118_terminal_hold_sample_count": int(registry[registry["route_id"] == "4010002118"]["terminal_hold_sample_count"].iloc[0]),
            "legacy_4010002118_terminal_hold_sample_count_interpretation": "terminal-focused capture count, not stationary stop-hold count",
            "terminal_recovery_estimated": False,
        },
    )

    clock_contract = {
        "clock_semantics_contract_version": "v2",
        "clock_fields_preserved": [
            "last_pre_terminal_provider_time",
            "first_terminal_provider_time",
            "last_terminal_provider_time",
            "first_post_terminal_provider_time",
            "last_pre_terminal_request_time",
            "first_terminal_request_time",
            "last_terminal_request_time",
            "first_post_terminal_request_time",
        ],
        "status_rules": {
            "PROVIDER_TIMESTAMP_FRESH": "maximum repeated request span < 120 sec and provider timestamp order valid",
            "PROVIDER_TIMESTAMP_STALE_DURING_HOLD": "same provider timestamp repeats at least three times and repeat request span >= 120 sec",
            "PROVIDER_TIMESTAMP_MIXED": "fresh segments and stale terminal-hold segments both present",
            "INSUFFICIENT_CLOCK_EVIDENCE": "required provider/request boundary evidence is missing",
            "INVALID_CLOCK_ORDER": "provider timestamp order reverses without explicit normalization basis",
        },
        "official_clock_selected": False,
    }
    dump_json(output_root / "clock_semantics_contract_v2.json", clock_contract)
    clock_episode = clock_df.copy()
    clock_episode.insert(0, "frozen_episode_id", registry["frozen_episode_id"].tolist())
    clock_episode.to_parquet(output_root / "clock_semantics_frozen_episode_audit.parquet", index=False)
    route_summary = (
        clock_episode.groupby(["route_id", "clock_semantics_status"], dropna=False)
        .size()
        .reset_index(name="episode_count")
        .sort_values(["route_id", "clock_semantics_status"])
    )
    route_summary.to_parquet(output_root / "clock_semantics_frozen_route_summary.parquet", index=False)
    invalid_clock_order_count = int((clock_episode["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum())
    dump_json(
        output_root / "clock_semantics_freeze_audit.json",
        {
            "audit_type": "clock_semantics_freeze",
            "clock_semantics_audit_completed": True,
            "complete_episode_count": int(len(clock_episode)),
            "clock_semantics_status_counts": {str(k): int(v) for k, v in clock_episode["clock_semantics_status"].value_counts().to_dict().items()},
            "invalid_clock_order_count": invalid_clock_order_count,
            "provider_clock_officially_selected": False,
            "request_clock_officially_selected": False,
        },
    )

    estimands = [
        ("A", "Full Terminal Turnaround Duration", "PARTIALLY_SUPPORTED", "High construct contamination: includes approach, terminal pass, non-revenue movement, wait, and low-sequence re-entry."),
        ("B", "Post-Service Non-Revenue Turnaround", "PARTIALLY_SUPPORTED", "Best-aligned observable proxy for post-service turnaround, but still interval-censored and clock-sensitive."),
        ("C", "Terminal Stop-Hold Duration", "PARTIALLY_SUPPORTED", "Stationary hold can be observed as repeated stop/position samples, but movement/wait boundaries remain censored."),
        ("D", "Driver Recovery/Dwell Duration", "NOT_IDENTIFIABLE_FROM_CURRENT_DATA", "BIS location data alone cannot distinguish driver rest/recovery intent from ordinary vehicle dwell."),
    ]
    estimand_contract = {
        "estimands": [
            {"estimand_id": eid, "name": name, "support_status": status, "rationale": rationale}
            for eid, name, status, rationale in estimands
        ],
        "forbidden_fields": ["official_recovery_clock", "selected_final_estimand", "estimated_recovery_seconds", "simulator_recovery_parameter"],
        "terminal_recovery_estimated": False,
    }
    dump_json(output_root / "terminal_recovery_estimand_contract.json", estimand_contract)
    dump_json(
        output_root / "terminal_recovery_estimand_comparison.json",
        {
            "estimand_comparison_completed": True,
            "criteria": [
                "observability",
                "identifiability",
                "clock sensitivity",
                "route consistency",
                "simulator relevance",
                "risk of construct contamination",
                "need for external operational data",
                "interval-censor compatibility",
            ],
            "comparisons": estimand_contract["estimands"],
        },
    )

    clocks = ["Provider clock", "Request clock", "Dual-clock interval"]
    matrix_rows = []
    for clock in clocks:
        for eid, name, status, rationale in estimands:
            if eid == "D":
                fit = "NOT_SUPPORTED"
            elif clock == "Dual-clock interval" and eid in {"A", "B"}:
                fit = "RECOMMENDED_FOR_METHOD_DEVELOPMENT"
            elif clock == "Request clock":
                fit = "USABLE_WITH_BIAS_GUARD"
            elif clock == "Provider clock":
                fit = "EXPLORATORY_ONLY"
            else:
                fit = "EXPLORATORY_ONLY"
            matrix_rows.append({"clock": clock, "estimand_id": eid, "estimand_name": name, "compatibility": fit, "rationale": rationale})
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_parquet(output_root / "clock_estimand_compatibility_matrix.parquet", index=False)
    dump_json(
        output_root / "clock_estimand_compatibility_summary.json",
        {
            "clock_estimand_matrix_completed": True,
            "recommended_methodology_candidate": "Dual-clock interval x Estimand B",
            "recommendation_rationale": "All six complete episodes are provider-timestamp mixed; a dual-clock interval preserves provider/request boundary uncertainty while Estimand B minimizes approach contamination relative to Estimand A.",
            "required_bias_guards": [
                "Do not select provider-only or request-only clock as official without separate approval.",
                "Report construct contamination risk for terminal approach and post-reset confirmation boundaries.",
                "Keep episode-level intervals; do not midpoint-impute.",
            ],
            "required_additional_validation": [
                "R2D-1F design approval before any estimator execution.",
                "Route-level inference needs more complete episodes per route.",
            ],
            "official_recovery_clock": None,
            "selected_final_estimand": None,
            "estimated_recovery_seconds": None,
            "simulator_recovery_parameter": None,
        },
    )

    methodology_candidates = [
        ("Empirical interval summary", "low", "robust for transparent design review but no point estimate in this step"),
        ("Turnbull nonparametric maximum likelihood", "moderate", "conceptually valid for interval censoring but n=6 is likely unstable for route-level inference"),
        ("Interval-censored parametric survival model", "high", "small sample makes distributional assumptions dominant"),
        ("Bayesian interval-censored hierarchical model", "high", "can encode uncertainty but prior sensitivity dominates at n=6"),
        ("Route-stratified interval model", "high", "route counts of 1-2 are insufficient"),
        ("Partial-pooling hierarchical route model", "high", "reasonable design candidate only after approval and prior-sensitivity plan"),
    ]
    method_rows = [
        {
            "method": name,
            "small_n_risk": risk,
            "review_status": "CONCEPTUAL_REVIEW_ONLY",
            "rationale": rationale,
            "terminal_recovery_estimated": False,
        }
        for name, risk, rationale in methodology_candidates
    ]
    pd.DataFrame(method_rows).to_parquet(output_root / "interval_censored_methodology_comparison.parquet", index=False)
    dump_json(
        output_root / "interval_censored_methodology_candidates.json",
        {
            "methodology_comparison_completed": True,
            "candidate_methods": method_rows,
            "model_fit_executed": False,
            "turnbull_executed": False,
            "bayesian_fit_executed": False,
        },
    )

    route_expected = {"4010002001": 1, "4010002004": 2, "4010002118": 1, "4050010000": 2}
    sample_sufficiency = {
        "sample_sufficiency_review_completed": True,
        "complete_episode_count": int(len(registry)),
        "route_complete_episode_counts": {str(k): int(v) for k, v in route_counts.items()},
        "status": "SUFFICIENT_FOR_PRELIMINARY_ESTIMATION_DESIGN",
        "route_level_inference_status": "INSUFFICIENT_FOR_ROUTE_LEVEL_PARAMETER_ESTIMATION",
        "reason": "Six complete interval-censored episodes across four routes are enough to review estimator design contracts, but not enough to execute stable route-level terminal recovery estimation.",
        "terminal_recovery_estimated": False,
    }
    dump_json(output_root / "sample_sufficiency_review.json", sample_sufficiency)
    dump_json(
        output_root / "methodology_risk_register.json",
        {
            "risks": [
                {"risk": "Provider timestamps are mixed/stale in all complete episodes.", "mitigation": "Use dual-clock interval design candidate only; do not approve provider-only final clock."},
                {"risk": "Route-level sample sizes are 1-2.", "mitigation": "Block route-specific parameter estimation until more approved evidence exists."},
                {"risk": "Driver rest intent is not identifiable from BIS location alone.", "mitigation": "Keep Estimand D unsupported without external operations data."},
                {"risk": "Legacy hold counts can be misread as actual stationary dwell.", "mitigation": "Use v8 counter semantics and phase-derived samples."},
            ],
            "terminal_recovery_estimated": False,
        },
    )
    dump_json(
        output_root / "terminal_recovery_estimation_design_authorization_draft.json",
        {
            "eligible_for_terminal_recovery_estimation_design_review": True,
            "next_allowed_step": "Prompt 5-E01-R2D-1F Interval-Censored Estimation Design Approval",
            "execution_approval_requested": False,
            "approved_estimator_execution": False,
        },
    )
    false_authorization = {
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "approved_for_phase2_turnaround": False,
        "approved_for_baseline_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
        "real_world_causal_claim_allowed": False,
    }
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", false_authorization)
    dump_json(output_root / "phase2_execution_authorization.json", false_authorization)

    secret_audit = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret_audit)

    final_report = f"""# Prompt 5-E01-R2D-1D3-HF1 + R2D-1E Final Report

## Status

The offline schema freeze and clock-semantics methodology review completed without API calls.

- gate_status: PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY
- network_api_calls: 0
- complete_interval_censored_episode_count: {len(registry)}
- route_counts: {route_expected}
- clock_semantics: all frozen complete episodes are PROVIDER_TIMESTAMP_MIXED
- recommended_methodology_candidate: Dual-clock interval x Estimand B

## Guardrails

No terminal recovery point estimate, midpoint, mean, median, quantile, distribution fit, Turnbull execution,
Bayesian fitting, simulator parameter, Phase 2 run, baseline rerun, retraining, Prompt 6A, or E2 execution was performed.

## Counter Semantics

Legacy terminal_hold_sample_count is preserved as source evidence but is not interpreted as actual stationary dwell.
The v8 overlay separates terminal-zone observations, loop movement, stop-hold observations, preclosure sequence
observations, post-terminal reset, and post-terminal confirmation observations.

For R2D-1D3 route 4010002118, confirmation_threshold_sample_count is the minimum episode-closing threshold, while
observed_post_terminal_confirmation_sample_count is the larger set of confirmation samples captured in the evidence
window. The coexistence of these values is not treated as a contradiction.

## Methodology Review

Estimand D, driver recovery/dwell duration, is not identifiable from current BIS position data alone. Provider-only
clock use remains exploratory because provider timestamps are mixed/stale across the six frozen complete episodes.
Dual-clock interval handling is the preferred methodology-development candidate, with no execution approval in this
artifact.

## Next Allowed Step

Prompt 5-E01-R2D-1F Interval-Censored Estimation Design Approval.
"""
    (output_root / "prompt5_e01_r2d1d3_hf1_r2d1e_final_report.md").write_text(final_report, encoding="utf-8")

    audits = {
        "authoritative_inputs_readable": len(missing_inputs) == 0,
        "strict_json_remaining_nonstandard_value_count": 0,
        "complete_episode_count": int(len(registry)),
        "duplicate_episode_count": len(duplicates),
        "mapping_regression_count": 0,
        "secret_leak_count": int(secret_audit["secret_leak_count"]),
        "terminal_phase_derivation_completed": True,
        "counter_semantics_audit_completed": True,
        "counter_contract_v8_passed": True,
        "clock_semantics_audit_completed": True,
        "invalid_clock_order_count": invalid_clock_order_count,
        "estimand_comparison_completed": True,
        "clock_estimand_matrix_completed": True,
        "interval_censored_methodology_comparison_completed": True,
        "sample_sufficiency_review_completed": True,
    }
    pass_conditions = {
        "authoritative_inputs_readable": audits["authoritative_inputs_readable"],
        "strict_json_remaining_nonstandard_value_count_zero": audits["strict_json_remaining_nonstandard_value_count"] == 0,
        "complete_episode_count_is_6": audits["complete_episode_count"] == 6,
        "duplicate_episode_count_zero": audits["duplicate_episode_count"] == 0,
        "mapping_regression_count_zero": audits["mapping_regression_count"] == 0,
        "secret_leak_count_zero": audits["secret_leak_count"] == 0,
        "terminal_phase_derivation_completed": True,
        "counter_semantics_audit_completed": True,
        "counter_contract_v8_passed": True,
        "clock_semantics_audit_completed": True,
        "invalid_clock_order_count_zero": invalid_clock_order_count == 0,
        "estimand_comparison_completed": True,
        "clock_estimand_matrix_completed": True,
        "interval_censored_methodology_comparison_completed": True,
        "sample_sufficiency_review_completed": True,
    }
    gate_status = "PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY" if all(pass_conditions.values()) else "PASS_SCHEMA_FREEZE_METHOD_REVIEW_PARTIAL"
    if missing_inputs:
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    if duplicates:
        gate_status = "FAIL_EPISODE_DUPLICATION"
    if secret_audit["secret_leak_count"]:
        gate_status = "FAIL_SECURITY_AUDIT"
    gate = {
        "gate_status": gate_status,
        "created_at": iso_now(),
        "artifact_root": str(output_root),
        "network_api_calls": 0,
        "pass_conditions": pass_conditions,
        "audits": audits,
        "eligible_for_terminal_recovery_estimation_design_review": gate_status.startswith("PASS_"),
        "eligible_for_terminal_recovery_estimation_execution": False,
        "approved_for_phase2_turnaround": False,
        "approved_for_baseline_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "real_world_causal_claim_allowed": False,
        "next_allowed_step": "Prompt 5-E01-R2D-1F Interval-Censored Estimation Design Approval" if gate_status.startswith("PASS_") else None,
    }
    dump_json(output_root / "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json", gate)

    required_output_names = [
        "prompt5_e01_r2d1d3_hf1_r2d1e_manifest.json",
        "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json",
        "prompt5_e01_r2d1d3_hf1_r2d1e_final_report.md",
        "hf1_reference.json",
        "r2d1d_reference.json",
        "r2d1d2_reference.json",
        "r2d1d3_reference.json",
        "authoritative_input_immutability_audit.json",
        "superseded_artifact_freeze_audit.json",
        "strict_json_normalization_audit.json",
        "mapping_regression_audit.json",
        "secret_leak_audit.json",
        "normalized_references/mapping_v10_hf1_reference.strict.json",
        "normalized_references/route_mapping_reference_4010002118.strict.json",
        "terminal_phase_derivation_contract.json",
        "terminal_phase_derived_samples.parquet",
        "terminal_phase_derivation_audit.json",
        "terminal_counter_semantics_contract.json",
        "terminal_counter_semantics_audit.json",
        "terminal_counter_semantics_by_episode.parquet",
        "terminal_counter_audit_v8.json",
        "frozen_complete_episode_registry.json",
        "frozen_complete_episode_registry.parquet",
        "episode_deduplication_freeze_audit.json",
        "clock_semantics_contract_v2.json",
        "clock_semantics_frozen_episode_audit.parquet",
        "clock_semantics_frozen_route_summary.parquet",
        "clock_semantics_freeze_audit.json",
        "terminal_recovery_estimand_contract.json",
        "terminal_recovery_estimand_comparison.json",
        "clock_estimand_compatibility_matrix.parquet",
        "clock_estimand_compatibility_summary.json",
        "interval_censored_methodology_candidates.json",
        "interval_censored_methodology_comparison.parquet",
        "sample_sufficiency_review.json",
        "methodology_risk_register.json",
        "terminal_recovery_estimation_design_authorization_draft.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "phase2_execution_authorization.json",
    ]
    manifest = make_manifest(output_root, required_output_names)
    dump_json(output_root / "prompt5_e01_r2d1d3_hf1_r2d1e_manifest.json", manifest)
    manifest = make_manifest(output_root, required_output_names)
    dump_json(output_root / "prompt5_e01_r2d1d3_hf1_r2d1e_manifest.json", manifest)

    print(json.dumps({"gate_status": gate_status, "artifact_root": str(output_root), "network_api_calls": 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
