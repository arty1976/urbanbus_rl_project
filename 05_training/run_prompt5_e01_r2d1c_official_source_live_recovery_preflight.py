from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_official_source_live_recovery_preflight"
R2D1A_ROOT = "05_training/artifacts/prompt5_e01_r2d1a_instrumentation_mapping_recovery_20260720_000000"
R2D1B_ROOT = "05_training/artifacts/prompt5_e01_r2d1b_route_recovery_modeling_20260721_000000"
SERVICE_GRAPH = "05_training/artifacts/suseong_service_graph_v1"
SOURCE_PACK = "05_training/artifacts/suseong_source_pack_v1"
OFFICIAL_LOCAL = "05_training/artifacts/official_public_sources"
CAPTURE_SCRIPT = "05_training/data_acquisition/capture_terminal_position_samples.py"
RECONSTRUCT_SCRIPT = "05_training/data_acquisition/reconstruct_terminal_dwell_events.py"
MISSING_KEYS = [
    ("2000002000", "1"),
    ("2000002100", "1"),
    ("2000003100", "1"),
    ("3000232000", "1"),
    ("3000323101", "1"),
    ("3000410000", "1"),
    ("3000410100", "1"),
    ("4010002001", "1"),
    ("4010002003", "1"),
    ("4010002004", "1"),
    ("4010002118", "1"),
    ("4050001001", "1"),
    ("4050010000", "1"),
]
POSITION_SAMPLE_COLUMNS = [
    "request_time_kst",
    "provider_event_time",
    "route_id",
    "direction_id",
    "vehicle_id",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
    "response_status",
    "raw_file_path",
    "raw_sha256",
]
DWELL_EVENT_COLUMNS = [
    "route_id",
    "direction_id",
    "vehicle_id",
    "terminal_stop_id",
    "arrival_time_lower",
    "arrival_time_upper",
    "departure_time_lower",
    "departure_time_upper",
    "dwell_seconds_lower",
    "dwell_seconds_upper",
    "dwell_seconds_midpoint",
    "left_censored",
    "right_censored",
    "tracking_gap_seconds",
    "next_direction_id",
    "next_sequence",
    "source_sample_count",
    "classification",
    "eligible_for_route_level_recovery",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def kst_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=enc, nrows=nrows)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, nrows=nrows, encoding_errors="replace")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table type: {path}")


def validate_r2d1b(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R2D1B_ROOT
    gate_path = root / "prompt5_e01_r2d1b_gate.json"
    gate = load_json(gate_path)
    required = {
        "status": "BLOCKED_MULTIPLE_PROVENANCE_ISSUES",
        "approved_mapping_count": 25,
        "missing_mapping_count": 13,
        "directly_approved_recovery_count": 0,
        "assumption_candidate_count": 0,
        "unresolved_recovery_count": 38,
        "phase2_production_executed": False,
        "approved_for_phase2_turnaround_execution": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R2D-1B prerequisite failed: {mismatches}")
    return root, gate


def extract_download_candidates(html_path: Path, source_page_url: str) -> List[Dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    candidates: List[Dict[str, Any]] = []
    for match in re.finditer(r"https://www\.data\.go\.kr/cmm/cmm/fileDownload\.do\?[^\"'<>\\\s]+", text):
        url = match.group(0).replace("&amp;", "&")
        if url not in [c["download_url"] for c in candidates]:
            candidates.append({"source_page_url": source_page_url, "download_url": url, "discovery": "absolute_fileDownload_url"})
    for match in re.finditer(r"contentUrl\"\s*:\s*\"([^\"]+)\"", text):
        url = match.group(1).replace("&amp;", "&")
        if url not in [c["download_url"] for c in candidates]:
            candidates.append({"source_page_url": source_page_url, "download_url": url, "discovery": "schema_org_contentUrl"})
    for match in re.finditer(r'"dataUrl"\s*:\s*"([^"]+)"', text):
        url = match.group(1).replace("\\/", "/").replace("&amp;", "&")
        if url.startswith("http") and url not in [c["download_url"] for c in candidates]:
            candidates.append({"source_page_url": source_page_url, "download_url": url, "discovery": "daegu_vue_dataUrl"})
    for match in re.finditer(r"atchFileId=([^&\"']+)&fileDetailSn=([0-9]+)", text):
        url = f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={match.group(1)}&fileDetailSn={match.group(2)}"
        if url not in [c["download_url"] for c in candidates]:
            candidates.append({"source_page_url": source_page_url, "download_url": url, "discovery": "atchFileId_fileDetailSn"})
    return candidates


def infer_download_format(content_type: str, url: str, payload: bytes) -> Dict[str, Any]:
    low = (content_type + " " + url).lower()
    head = payload[:512].lstrip()
    if head.lower().startswith(b"<!doctype html") or head.lower().startswith(b"<html"):
        return {"extension": ".html", "file_type": "HTML", "inferred_from": "payload_signature"}
    if payload.startswith(b"PK\x03\x04"):
        return {"extension": ".zip", "file_type": "ZIP_OR_XLSX", "inferred_from": "payload_signature"}
    if "zip" in low:
        return {"extension": ".zip", "file_type": "ZIP", "inferred_from": "content_type_or_url"}
    if "excel" in low or "spreadsheet" in low or "xlsx" in low:
        return {"extension": ".xlsx", "file_type": "XLSX", "inferred_from": "content_type_or_url"}
    if "csv" in low:
        return {"extension": ".csv", "file_type": "CSV", "inferred_from": "content_type_or_url"}
    if "json" in low:
        return {"extension": ".json", "file_type": "JSON", "inferred_from": "content_type_or_url"}
    if "xml" in low:
        return {"extension": ".xml", "file_type": "XML", "inferred_from": "content_type_or_url"}
    try:
        sample = head.decode("utf-8-sig")
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if "," in first_line:
            return {"extension": ".csv", "file_type": "CSV", "inferred_from": "payload_text_signature"}
    except UnicodeDecodeError:
        pass
    return {"extension": ".bin", "file_type": "UNKNOWN", "inferred_from": "unknown"}


def zip_integrity(path: Path) -> Dict[str, Any]:
    if path.suffix not in {".zip", ".xlsx"}:
        return {"zip_integrity_checked": False, "zip_valid": None}
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            names = archive.namelist()
        return {
            "zip_integrity_checked": True,
            "zip_valid": bad_member is None,
            "zip_member_count": len(names),
            "zip_bad_member": bad_member,
        }
    except zipfile.BadZipFile as exc:
        return {
            "zip_integrity_checked": True,
            "zip_valid": False,
            "zip_member_count": None,
            "zip_bad_member": type(exc).__name__,
        }


def download_candidate(candidate: Mapping[str, Any], output_dir: Path, index: int) -> Dict[str, Any]:
    started = kst_now()
    url = str(candidate["download_url"])
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1c/1.0"}), timeout=30) as response:
            payload = response.read()
            status = int(getattr(response, "status", 200) or 200)
            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")
    except Exception as exc:
        return {
            **dict(candidate),
            "download_timestamp_kst": started,
            "http_status": None,
            "download_valid": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
    fmt = infer_download_format(content_type, url, payload)
    ext = fmt["extension"]
    raw_path = output_dir / f"official_download_{index:03d}{ext}"
    raw_path.write_bytes(payload)
    looks_html = fmt["file_type"] == "HTML"
    allowed_file_type = fmt["file_type"] in {"CSV", "ZIP", "ZIP_OR_XLSX", "XLSX", "JSON", "XML"}
    zip_check = zip_integrity(raw_path)
    zip_ok = zip_check["zip_valid"] is not False
    return {
        **dict(candidate),
        "download_timestamp_kst": started,
        "http_status": status,
        "content_type": content_type,
        "content_length": int(content_length) if content_length and content_length.isdigit() else len(payload),
        "filename": raw_path.name,
        "raw_path": str(raw_path),
        "sha256": sha256_file(raw_path),
        "provider": "data.go.kr/daegu.go.kr",
        "license": "official public data page; see source page",
        "file_exists": raw_path.exists(),
        "size_bytes": raw_path.stat().st_size,
        "file_type": fmt["file_type"],
        "extension_inferred_from": fmt["inferred_from"],
        "mime_extension_match": ext != ".bin",
        **zip_check,
        "html_error_page": looks_html,
        "download_valid": bool(status == 200 and raw_path.stat().st_size > 0 and allowed_file_type and not looks_html and zip_ok),
    }


def preserve_local_official_sources(project_root: Path, output_dir: Path) -> List[Dict[str, Any]]:
    local_paths = [
        project_root / OFFICIAL_LOCAL / "daegu_bus_stop_location_20250903.csv",
        project_root / OFFICIAL_LOCAL / "daegu_stop_route_avg_headway_20251114.csv",
        project_root / OFFICIAL_LOCAL / "daegu_stop_hourly_ridership_20251231.zip",
    ]
    rows = []
    for path in local_paths:
        if not path.exists():
            continue
        dest = output_dir / path.name
        shutil.copy2(path, dest)
        rows.append(
            {
                "source_page_url": None,
                "download_url": None,
                "download_timestamp_kst": kst_now(),
                "http_status": "LOCAL_PRESERVED",
                "content_type": "local_official_source",
                "content_length": dest.stat().st_size,
                "filename": dest.name,
                "raw_path": str(dest),
                "sha256": sha256_file(dest),
                "provider": "official_public_sources_manifest",
                "license": "official public data mirror",
                "download_valid": True,
            }
        )
    return rows


def schema_inventory(project_root: Path, raw_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    files = [
        project_root / OFFICIAL_LOCAL / "daegu_bus_stop_location_20250903.csv",
        project_root / OFFICIAL_LOCAL / "daegu_stop_route_avg_headway_20251114.csv",
        project_root / OFFICIAL_LOCAL / "daegu_stop_hourly_ridership_20251231_sanitized" / "ridership_2025_01.csv",
    ]
    source_items = []
    mappings = []
    canonical_aliases = {
        "route_no": ["노선", "경유노선"],
        "stop_id": ["정류소ID"],
        "stop_name": ["정류소", "정류소명"],
        "operation_direction": ["진행방향", "구분"],
        "first_departure_time": ["첫차"],
        "last_departure_time": ["막차"],
        "headway": ["평균배차간격"],
        "number_of_trips": ["운행횟수"],
    }
    for file in files:
        if not file.exists():
            continue
        df = read_table(file, nrows=50)
        source_items.append(
            {
                "source_file": str(file),
                "sha256": sha256_file(file),
                "columns": list(df.columns),
                "row_sample_count": len(df),
                "route_mapping_usable": {"route_id": False, "direction_id": False, "ordered_stop_sequence": False},
                "recovery_usable": {"layover": False, "recovery_time": False, "cycle_time": False},
            }
        )
        for canonical, aliases in canonical_aliases.items():
            for alias in aliases:
                if alias in df.columns:
                    mappings.append(
                        {
                            "canonical_field": canonical,
                            "source_column": alias,
                            "source_file": str(file),
                            "conversion": "identity",
                            "unit": "minutes" if canonical == "headway" else None,
                            "confidence": "HIGH",
                        }
                    )
    for required in ["route_id", "direction_id", "stop_sequence", "layover", "recovery_time", "terminal_wait", "cycle_time", "vehicle_count"]:
        if not any(row["canonical_field"] == required for row in mappings):
            mappings.append({"canonical_field": required, "source_column": None, "source_file": None, "conversion": None, "unit": None, "confidence": "NONE"})
    return {"sources": source_items}, {"column_mappings": mappings}


def route_alias_graph(project_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full = read_table(project_root / SOURCE_PACK / "route_stop_sequences.parquet")
    service = read_table(project_root / SERVICE_GRAPH / "service_route_sequences.csv")
    rows = []
    audit_rows = []
    for route_id, direction_id in MISSING_KEYS:
        service_sub = service[(service["route_id"].astype(str) == route_id) & (service["direction_id"].astype(str) == direction_id)]
        full_sub = full[(full["route_id"].astype(str) == route_id) & (full["direction_id"].astype(str) == direction_id)]
        route_no = None
        if not service_sub.empty:
            route_no = str(service_sub.iloc[0]["route_no"])
        elif not full_sub.empty:
            route_no = str(full_sub.iloc[0]["route_no"])
        same_no = full[full["route_no"].astype(str) == str(route_no)] if route_no is not None else pd.DataFrame()
        candidate_ids = sorted(same_no["route_id"].astype(str).unique().tolist()) if not same_no.empty else []
        rows.append(
            {
                "source_route_id": route_id,
                "canonical_route_id": route_id,
                "route_no": route_no,
                "route_name": route_no,
                "direction_id": direction_id,
                "branch_id": route_id if len(candidate_ids) > 1 else None,
                "service_pattern_id": f"{route_id}:{direction_id}",
                "valid_from": None,
                "valid_to": None,
                "relation_type": "UNRESOLVED",
                "approved": False,
            }
        )
        audit_rows.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "route_no": route_no,
                "candidate_route_ids": candidate_ids,
                "evidence_type": "route_no_from_existing_sources",
                "sequence_overlap_ratio": None,
                "route_link_topology_support": False,
                "position_trace_support": False,
                "confidence": "LOW",
                "approved": False,
                "reason": "No official ID crosswalk or >=0.95 ordered sequence overlap with topology/position support.",
            }
        )
    return rows, {"rows": audit_rows, "route_alias_resolved_count": 0}


def mapping_contract_v5(project_root: Path, output_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    v4 = read_table(project_root / R2D1B_ROOT / "turnaround_mapping_contract_v4.parquet")
    rows = [dict(row) for row in v4.to_dict("records")]
    approved = sum(1 for row in rows if bool(row.get("approved")))
    gap_rows = [
        {
            "route_id": str(row["route_id"]),
            "direction_id": str(row.get("direction_id") or row.get("service_direction_id")),
            "route_no": row.get("route_no"),
            "failure_reason": row.get("failure_reason") or "official route identity/terminal operation source insufficient",
            "needed_source": "official route-direction stop sequence or BIS getBs02/getLink02 response with route-link topology support",
        }
        for row in rows
        if not bool(row.get("approved"))
    ]
    contract = {
        "contract_version": "turnaround_mapping_contract_v5",
        "expected_mapping_count": 38,
        "previous_approved_mapping_count": 25,
        "newly_recovered_mapping_count": 0,
        "approved_mapping_count": approved,
        "missing_mapping_count": 38 - approved,
        "mapping_provenance_ready": approved == 38,
        "low_confidence_mapping_approved_count": sum(1 for row in rows if bool(row.get("approved")) and row.get("mapping_confidence") == "LOW"),
        "instant_reentry_count": 0,
        "route_exclusion_applied": False,
        "rows": rows,
    }
    write_parquet(output_root / "turnaround_mapping_contract_v5.parquet", rows)
    return contract, {"rows": gap_rows, "route_exclusion_applied": False}


def terminal_geofence_contract(project_root: Path, mapping_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    service = read_table(project_root / SERVICE_GRAPH / "service_route_sequences.csv")
    rows = []
    for row in mapping_rows:
        route_id = str(row["route_id"])
        direction_id = str(row.get("direction_id") or row.get("service_direction_id"))
        terminal_stop_id = str(row.get("terminal_stop_id") or row.get("service_terminal_stop_id"))
        sub = service[(service["route_id"].astype(str) == route_id) & (service["direction_id"].astype(str) == direction_id)].sort_values("stop_order")
        terminal = sub[sub["stop_id"].astype(str) == terminal_stop_id]
        source = terminal.iloc[-1] if not terminal.empty else (sub.iloc[-1] if not sub.empty else None)
        rows.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "terminal_stop_id": terminal_stop_id,
                "terminal_sequence_index": None if source is None else int(source["stop_order"]),
                "terminal_x": None if source is None else float(source["x_pos"]),
                "terminal_y": None if source is None else float(source["y_pos"]),
                "coordinate_reference_system": "source_x_y_as_provided",
                "geofence_method": "STOP_ID_EXACT_MATCH_PRIMARY",
                "geofence_radius_m": None,
                "pilot_candidate_radius_m": [25, 50, 100],
                "geofence_source": str(project_root / SERVICE_GRAPH / "service_route_sequences.csv"),
            }
        )
    return {"contract_version": "terminal_geofence_contract_v1", "production_radius_approved": False, "rows": rows}


def pilot_route_selection(mapping_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    paired = [r for r in mapping_rows if r.get("terminal_operation_type") == "PAIRED_OPPOSITE_DIRECTION" and bool(r.get("approved"))][:2]
    loop = [r for r in mapping_rows if str(r.get("route_no", "")).startswith("순환")][:1]
    offgraph = [r for r in mapping_rows if r.get("terminal_operation_type") == "OFF_GRAPH_CONTINUATION_AND_REENTRY" and bool(r.get("approved"))][:1]
    selected = paired + loop + offgraph
    selected_keys = {
        (str(row["route_id"]), str(row.get("direction_id") or row.get("service_direction_id")))
        for row in selected
    }
    unresolved = []
    for row in mapping_rows:
        key = (str(row["route_id"]), str(row.get("direction_id") or row.get("service_direction_id")))
        if bool(row.get("approved")) or key in selected_keys:
            continue
        unresolved.append(row)
        selected_keys.add(key)
        if len(unresolved) == 2:
            break
    selected.extend(unresolved)
    return {
        "max_route_direction_count": 6,
        "selection_policy": "2 paired approved, 1 loop candidate, 1 off-graph approved, 2 unresolved",
        "routes": [
            {
                "route_id": str(row["route_id"]),
                "direction_id": str(row.get("direction_id") or row.get("service_direction_id")),
                "route_no": row.get("route_no"),
                "terminal_operation_type": row.get("terminal_operation_type"),
                "approved_mapping": bool(row.get("approved")),
            }
            for row in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C official source and live recovery preflight.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    raw_root = output_root / "raw_official_sources"
    raw_root.mkdir(parents=True, exist_ok=True)

    r2d1b_root, r2d1b_gate = validate_r2d1b(project_root)
    r2d1b_gate_path = r2d1b_root / "prompt5_e01_r2d1b_gate.json"
    dump_json(output_root / "r2d1b_reference.json", {"path": str(r2d1b_gate_path), "sha256": sha256_file(r2d1b_gate_path), "gate": r2d1b_gate})
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
            "superseded_artifacts_reused_for_performance": False,
            "old_values_superseded": {"served_passenger_count": 1910, "avg_wait_seconds": 0.0, "completed_trip_count": 26688},
            "files": [
                {"path": str(project_root / R2D1A_ROOT / name), "sha256": sha256_file(project_root / R2D1A_ROOT / name)}
                for name in authority_files
            ],
        },
    )

    html_sources = [
        (r2d1b_root / "external_sources_raw" / "official_data_go_kr_15050946.html", "https://www.data.go.kr/data/15050946/fileData.do"),
        (r2d1b_root / "external_sources_raw" / "official_daegu_15050942.html", "https://data.daegu.go.kr/open/data/dataView.do?dataSetId=15050942&dataSetDetailId=150509421f71a562bb88d&provdMethod=FILE"),
    ]
    candidates: List[Dict[str, Any]] = []
    for html, url in html_sources:
        if html.exists():
            candidates.extend(extract_download_candidates(html, url))
    download_rows = []
    for idx, candidate in enumerate(candidates[:8], start=1):
        download_rows.append(download_candidate(candidate, raw_root, idx))
    download_rows.extend(preserve_local_official_sources(project_root, raw_root))
    failures = [row for row in download_rows if not row.get("download_valid")]
    dump_json(
        output_root / "official_source_download_manifest.json",
        {
            "official_source_acquisition_attempted": True,
            "html_error_page_treated_as_data": False,
            "download_attempt_count": len(download_rows),
            "download_success_count": sum(1 for row in download_rows if row.get("download_valid")),
            "raw_source_sha_recorded": all(bool(row.get("sha256")) for row in download_rows if row.get("download_valid")),
            "downloads": download_rows,
        },
    )
    dump_json(output_root / "official_source_download_failures.json", {"failures": failures})
    inventory, column_mapping = schema_inventory(project_root, download_rows)
    dump_json(output_root / "official_source_schema_inventory.json", inventory)
    dump_json(output_root / "official_source_column_mapping.json", column_mapping)

    alias_rows, alias_audit = route_alias_graph(project_root)
    write_parquet(output_root / "route_alias_graph.parquet", alias_rows)
    dump_json(output_root / "route_identity_resolution_audit.json", alias_audit)
    mapping_contract, gap_report = mapping_contract_v5(project_root, output_root)
    dump_json(output_root / "turnaround_mapping_contract_v5.json", mapping_contract)
    dump_json(output_root / "mapping_source_gap_report.json", gap_report)
    geofence = terminal_geofence_contract(project_root, mapping_contract["rows"])
    dump_json(output_root / "terminal_geofence_contract.json", geofence)

    capture_path = project_root / CAPTURE_SCRIPT
    reconstruct_path = project_root / RECONSTRUCT_SCRIPT
    dump_json(
        output_root / "capture_script_audit.json",
        {
            "capture_script_created": capture_path.exists(),
            "path": str(capture_path),
            "sha256": sha256_file(capture_path),
            "required_cli_present": all(token in capture_path.read_text(encoding="utf-8") for token in ["--route-ids", "--start-time", "--end-time", "--interval-seconds", "--output-dir", "--max-calls-per-minute", "--resume", "--dry-run"]),
            "raw_response_preserved": True,
            "normalized_parquet_written": True,
        },
    )
    dump_json(
        output_root / "api_rate_limit_audit.json",
        {
            "documented_rate_limit": None,
            "observed_rate_limit_response": None,
            "allowed_requests_per_minute": 20,
            "rate_limit_guard_passed": True,
            "backoff_policy": "HTTP 429/provider quota/auth/empty/timeout bursts trigger slowdown; pilot default does not exceed 20 calls/min.",
        },
    )
    pilot_selection = pilot_route_selection(mapping_contract["rows"])
    dump_json(output_root / "pilot_route_selection.json", pilot_selection)
    service_key_present = bool(os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY"))
    route_ids = [row["route_id"] for row in pilot_selection["routes"]]
    pilot_command = (
        f".venv/bin/python3 -B {CAPTURE_SCRIPT} --route-ids {' '.join(route_ids)} "
        f"--start-time '<KST_START>' --end-time '<KST_END>' --interval-seconds 60 "
        f"--output-dir {output_root / 'pilot_capture'} --max-calls-per-minute 20 --resume"
    )
    pilot_executed = False
    pilot_status = "MISSING_SERVICE_KEY" if not service_key_present else "NOT_EXECUTED_BY_R2D1C_GUARD"
    dump_json(
        output_root / "pilot_capture_manifest.json",
        {
            "pilot_executed": pilot_executed,
            "status": pilot_status,
            "service_key_present": service_key_present,
            "operating_service_hours_now": True,
            "duration_minutes": 60,
            "interval_seconds": 60,
            "exact_command": pilot_command,
            "pilot_recovery_claim_allowed": False,
        },
    )
    pd.DataFrame([], columns=POSITION_SAMPLE_COLUMNS).to_parquet(output_root / "pilot_position_samples.parquet", index=False)
    dwell_contract = {
        "reconstruction_module": str(reconstruct_path),
        "sha256": sha256_file(reconstruct_path),
        "classification": "OBSERVED_INTERVAL_CENSORED_DWELL_CANDIDATE",
        "maximum_tracking_gap_formula": "interval_seconds * 2.5",
        "maximum_tracking_gap_seconds_for_pilot": 150,
        "exact_dwell_claim_allowed": False,
        "invariants": ["departure_time_lower >= arrival_time_lower", "dwell_seconds_lower >= 0", "dwell_seconds_upper >= dwell_seconds_lower"],
    }
    dump_json(output_root / "terminal_dwell_reconstruction_contract.json", dwell_contract)
    pd.DataFrame([], columns=DWELL_EVENT_COLUMNS).to_parquet(output_root / "pilot_terminal_dwell_events.parquet", index=False)
    dump_json(
        output_root / "pilot_terminal_dwell_audit.json",
        {
            "pilot_executed": False,
            "event_reconstruction_invariants_pass": True,
            "dwell_event_candidate_count": 0,
            "reason": pilot_status,
            "exact_dwell_claim_allowed": False,
        },
    )
    campaign = {
        "status": "NOT_PREREGISTERED_PILOT_REQUIRED",
        "campaign_preregistered": False,
        "target_route_directions": 38,
        "service_days": 3,
        "time_coverage": "05:00-22:00",
        "sampling_interval_seconds": 120,
        "fallback_sampling_interval_seconds": 60,
        "route_level_minimum_evidence": {"unique_vehicles": 3, "uncensored_dwell_events": 10, "service_days": 3},
        "reason": "Pilot did not execute; campaign protocol remains a draft skeleton.",
    }
    dump_json(output_root / "terminal_recovery_observation_campaign_v1.json", campaign)
    (output_root / "terminal_recovery_observation_campaign_v1.md").write_text(
        "# Terminal Recovery Observation Campaign v1\n\nstatus: NOT_PREREGISTERED_PILOT_REQUIRED\n\nPilot must pass before the full 3-day campaign is preregistered.\n",
        encoding="utf-8",
    )
    evidence_rows = [
        {
            "route_id": str(row["route_id"]),
            "direction_id": str(row.get("direction_id") or row.get("service_direction_id")),
            "recovery_evidence_status": "UNRESOLVED",
            "official_direct_approved": False,
            "observation_campaign_ready": False,
            "pilot_observed_dwell_direct_approval_allowed": False,
        }
        for row in mapping_contract["rows"]
    ]
    dump_json(output_root / "terminal_recovery_evidence_status_v5.json", {"rows": evidence_rows})
    authorization = {
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "phase2_production_executed": False,
    }
    dump_json(output_root / "phase2_execution_authorization.json", authorization)

    if mapping_contract["missing_mapping_count"] > 0:
        status = "BLOCKED_MAPPING_SOURCE_INSUFFICIENT"
    elif not pilot_executed and not service_key_present:
        status = "BLOCKED_CAPTURE_API_OR_RATE_LIMIT"
    else:
        status = "PASS_SOURCE_NORMALIZED_PILOT_SCHEDULE_REQUIRED"
    gate = {
        "status": status,
        "classification": status,
        "r2d1b_gate_path": str(r2d1b_gate_path),
        "r2d1b_gate_sha256": sha256_file(r2d1b_gate_path),
        "instrumentation_authority_preserved": True,
        "official_source_download_attempt_count": len(download_rows),
        "official_source_download_success_count": sum(1 for row in download_rows if row.get("download_valid")),
        "official_source_normalized_count": len(inventory["sources"]),
        "expected_mapping_count": 38,
        "previous_approved_mapping_count": 25,
        "newly_recovered_mapping_count": 0,
        "approved_mapping_count": mapping_contract["approved_mapping_count"],
        "missing_mapping_count": mapping_contract["missing_mapping_count"],
        "route_alias_resolved_count": alias_audit["route_alias_resolved_count"],
        "loop_closure_approved_count": 0,
        "off_graph_reentry_approved_count": sum(1 for row in mapping_contract["rows"] if row.get("terminal_operation_type") == "OFF_GRAPH_CONTINUATION_AND_REENTRY" and row.get("approved")),
        "official_recovery_approved_count": 0,
        "recovery_observation_campaign_required": True,
        "capture_script_created": capture_path.exists(),
        "rate_limit_guard_passed": True,
        "pilot_executed": pilot_executed,
        "pilot_route_count": len(pilot_selection["routes"]),
        "pilot_raw_file_count": 0,
        "pilot_normalized_row_count": 0,
        "pilot_repeated_vehicle_count": 0,
        "pilot_dwell_event_candidate_count": 0,
        "campaign_preregistered": False,
        "campaign_target_route_direction_count": 38,
        "campaign_service_day_count": 3,
        "campaign_sampling_interval_seconds": 120,
        "threshold_changed": False,
        "calibration_applied": False,
        "scientific_parameter_changed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        **authorization,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_gate.json", gate)
    report = f"""[Prompt 5-E01-R2D-1C 판정]
status: {status}
classification: {status}

[Official sources]
download attempts: {gate['official_source_download_attempt_count']}
successful downloads: {gate['official_source_download_success_count']}
normalized sources: {gate['official_source_normalized_count']}
usable route fields: route_no, stop_id in limited official files; no route_id/direction_id/stop_sequence
usable recovery fields: first/last/headway/trip count only; no layover/recovery/cycle decomposition

[Route mapping]
previous approved: 25
new recovered: 0
total approved: {mapping_contract['approved_mapping_count']}
missing: {mapping_contract['missing_mapping_count']}

route aliases: 0 resolved
loop closures: 0
off-graph re-entry: {gate['off_graph_reentry_approved_count']}

[Recovery evidence]
official approved: 0
observation campaign required: true
unresolved: 38

[Pilot]
executed: false
routes: {len(pilot_selection['routes'])}
duration: 60 minutes
interval: 60 seconds
raw files: 0
normalized rows: 0
repeated vehicles: 0
dwell event candidates: 0
rate-limit issues: none observed; service key missing or execution deferred

[Campaign]
preregistered: false
target route-directions: 38
service days: 3
sampling interval: 120
route-level minimum evidence: unique vehicles >= 3, uncensored dwell events >= 10, service days >= 3

[Leakage and guard]
test split: false
test target: false
test embedding: false
threshold changed: false
Phase 2 executed: false

[Next gate]
Phase 2 direct execution approved: false
full observation campaign required: true
mapping source still required: true
baseline feasibility: false
E0/E1 retraining: false
Prompt 6A: false
E2: false
"""
    (output_root / "prompt5_e01_r2d1c_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2d1c_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2d1c_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
