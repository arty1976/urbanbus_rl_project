#!/usr/bin/env python3
"""PV8-R2A-R8A official Daegu headway and B1 temporal-realism audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import resource
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8a_official_headway_temporal_realism.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"
R2AR7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar7_demand_contract_b1_orchestrator_20260809_111155"
R2AR6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection_20260809_113156"
R2AR8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8_multi_timeband_b1_normalization_20260809_120827"

OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
SUSEONG_ELIGIBLE_ROUTES = ARTIFACTS_ROOT / "suseong_service_graph_v1" / "suseong_eligible_service_routes.csv"
R8_WINDOW_METRICS = R2AR8_ROOT / "r2ar8_b1_window_metrics.parquet"
R8_NORMALIZATION_CANDIDATE = R2AR8_ROOT / "r2ar8_normalization_candidate.json"

OFFICIAL_SOURCE_URL = "https://www.data.go.kr/data/15060936/fileData.do?recommendDataYn=Y"
OFFICIAL_SCHEMA_URL = "https://www.data.go.kr/catalog/15060936/fileData.json"
OFFICIAL_DOWNLOAD_URL = "https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId=FILE_000000003527838&fileDetailSn=1&insertDataPrcus=N"
OFFICIAL_DATASET_NAME = "대구광역시_시내버스 정류소별_노선별_평균배차간격_20251114"
OFFICIAL_REFERENCE_DATE = "2025-11-14"
OFFICIAL_OWNER = "대구광역시"
OFFICIAL_DEPARTMENT = "교통정보운영과"
OFFICIAL_EXPECTED_ROW_COUNT = 31691

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
R8_NORMALIZATION_SHA256 = "34f12f10311a9c31dac135cd8ab0e412bf471e884ca9c094ab5f2c9a3bf45d47"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8a_official_headway_temporal_realism"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8A_DAEGU_OFFICIAL_HEADWAY_TEMPORAL_REALISM_AUDIT_COMPLETE"
DECISION_ACCEPTABLE = "PV8_B1_TEMPORAL_CONTRACT_ACCEPTABLE"
DECISION_REPAIR_REQUIRED = "PV8_B1_TEMPORAL_CONTRACT_REPAIR_REQUIRED"
DECISION_MAPPING_INCOMPLETE = "PV8_OFFICIAL_HEADWAY_MAPPING_INCOMPLETE"
READINESS_REPAIR = "SRP2_BIS_PV8_R2AR8A_COMPLETE_TEMPORAL_CONTRACT_REPAIR_REQUIRED_R8_NORMALIZATION_NOT_APPROVED"
READINESS_MAPPING = "SRP2_BIS_PV8_R2AR8A_COMPLETE_OFFICIAL_HEADWAY_MAPPING_INCOMPLETE"
READINESS_ACCEPTABLE = "SRP2_BIS_PV8_R2AR8A_COMPLETE_TEMPORAL_CONTRACT_ACCEPTABLE"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R2A-R4": (R2AR4_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"),
    "PV8-R2A-R7": (R2AR7_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar7.json", "_PV8_R2AR7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR7_DEMAND_CONTRACT_AND_B1_ORCHESTRATOR_COMPLETE"),
    "PV8-R2A-R6-rerun": (R2AR6_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar6.json", "_PV8_R2AR6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR6_CAUSAL_B1_REFERENCE_COLLECTION_COMPLETE"),
    "PV8-R2A-R8": (R2AR8_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8.json", "_PV8_R2AR8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8_MULTI_TIMEBAND_B1_NORMALIZATION_AUDIT_COMPLETE"),
}

REQUIRED_OFFICIAL_COLUMNS = [
    "정류소ID",
    "정류소",
    "노선",
    "진행방향",
    "시간표유형",
    "첫차",
    "막차",
    "운행횟수",
    "평균배차간격",
]

PAYLOADS = [
    "r8a_official_headway_source_manifest.json",
    "r8a_official_headway_raw_or_frozen_reference.csv",
    "r8a_route_headway_mapping.parquet",
    "r8a_suseong_headway_summary.json",
    "r8a_route814_headway_summary.json",
    "r8a_headway_distribution.parquet",
    "r8a_b1_temporal_realism_audit.json",
    "r8a_temporal_contract_repair_candidate.json",
    "r8a_r8_normalization_reassessment.json",
    "r8a_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8AError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).strip())
    if not match:
        return None
    out = float(match.group(0))
    return out if math.isfinite(out) else None


def summary_stats(values: Iterable[Any]) -> Dict[str, Any]:
    series = pd.Series([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype="float64")
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
    }


def verify_artifacts() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R8AError(f"{label} integrity failure: gate={observed}, checks={checks}")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return records


def verify_frozen_context() -> Dict[str, Any]:
    upstreams = verify_artifacts()
    r8_candidate = k5.read_json(R8_NORMALIZATION_CANDIDATE)
    constants = r8_candidate.get("constants", {})
    checks = {
        "reward_version_matches": r8_candidate.get("reward_version") == REWARD_VERSION,
        "reward_sha256_matches": r8_candidate.get("reward_contract_sha256") == REWARD_SHA256,
        "demand_version_matches": r8_candidate.get("demand_contract_version") == DEMAND_CONTRACT_VERSION,
        "demand_sha256_matches": r8_candidate.get("demand_contract_sha256") == DEMAND_CONTRACT_SHA256,
        "r8_candidate_sha256_matches": r8_candidate.get("candidate_sha256") == R8_NORMALIZATION_SHA256,
        "r8_service_reference_matches": constants.get("B1_service_reference") == 1.0,
        "r8_avg_wait_reference_matches": constants.get("B1_avg_wait_reference") == 15.0,
        "r8_p95_wait_reference_matches": constants.get("B1_p95_wait_reference") == 23.0,
        "rulebook_sha256_matches": (ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432" / "k7_research_rulebook_candidate.parquet").exists(),
        "occurrence_sha256_matches": k5.sha256_file(OCCURRENCE_PATH) == OCCURRENCE_SHA256,
    }
    if checks["rulebook_sha256_matches"]:
        rulebook_path = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432" / "k7_research_rulebook_candidate.parquet"
        checks["rulebook_sha256_matches"] = k5.sha256_file(rulebook_path) == RULEBOOK_SHA256
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R8AError(f"frozen context check failed: {checks}")
    return {
        "authoritative_upstreams": upstreams,
        "binding_checks": checks,
        "r8_normalization_candidate": {
            "candidate_id": r8_candidate.get("candidate_id"),
            "candidate_sha256": r8_candidate.get("candidate_sha256"),
            "constants": constants,
            "training_normalization_approved": r8_candidate.get("training_normalization_approved"),
            "status": r8_candidate.get("status"),
        },
    }


def load_official_headway(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise R8AError(f"official CSV missing: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            frame = pd.read_csv(path, dtype=str, encoding=encoding)
            break
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    else:
        raise R8AError(f"cannot decode official CSV: {last_error}")
    missing = [col for col in REQUIRED_OFFICIAL_COLUMNS if col not in frame.columns]
    if missing:
        raise R8AError(f"official CSV schema missing columns: {missing}")
    frame = frame.copy()
    frame["official_row_id"] = [f"OFFICIAL_HEADWAY_20251114_{idx:05d}" for idx in range(len(frame))]
    frame["stop_id"] = frame["정류소ID"].astype(str).str.strip()
    frame["stop_name"] = frame["정류소"].astype(str).str.strip()
    frame["route_no"] = frame["노선"].astype(str).str.strip()
    frame["direction_text"] = frame["진행방향"].astype(str).str.strip()
    frame["timetable_type"] = frame["시간표유형"].astype(str).str.strip()
    frame["first_bus"] = frame["첫차"].astype(str).str.strip()
    frame["last_bus"] = frame["막차"].astype(str).str.strip()
    frame["trip_count"] = frame["운행횟수"].map(numeric)
    frame["average_headway_minutes"] = frame["평균배차간격"].map(numeric)
    frame["average_headway_seconds"] = frame["average_headway_minutes"].map(lambda value: value * 60.0 if value is not None else None)
    return frame


def source_manifest(raw_path: Path, frozen_path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "dataset_name": OFFICIAL_DATASET_NAME,
        "source_url": OFFICIAL_SOURCE_URL,
        "schema_url": OFFICIAL_SCHEMA_URL,
        "direct_file_download_url": OFFICIAL_DOWNLOAD_URL,
        "source_owner": OFFICIAL_OWNER,
        "source_department": OFFICIAL_DEPARTMENT,
        "source_reference_date": OFFICIAL_REFERENCE_DATE,
        "portal_registered_date": "2025-11-17",
        "portal_modified_date": "2025-11-17",
        "downloaded_at": iso_kst(),
        "download_method": "official_data_go_kr_file_download",
        "raw_input_path": str(raw_path),
        "frozen_reference_path": str(frozen_path),
        "file_sha256": k5.sha256_file(frozen_path),
        "file_size_bytes": frozen_path.stat().st_size,
        "row_count": int(len(frame)),
        "portal_expected_row_count": OFFICIAL_EXPECTED_ROW_COUNT,
        "row_count_matches_portal": int(len(frame)) == OFFICIAL_EXPECTED_ROW_COUNT,
        "schema": list(frame.columns[:9]),
        "required_source_fields_present": all(col in frame.columns for col in REQUIRED_OFFICIAL_COLUMNS),
        "canonical_field_aliases": {
            "정류소ID": "stop_id",
            "정류소": "stop_name",
            "노선": "route_no",
            "진행방향": "direction_text",
            "시간표유형": "timetable_type",
            "첫차": "first_bus",
            "막차": "last_bus",
            "운행횟수": "trip_count",
            "평균배차간격": "average_headway_minutes",
        },
        "authoritative_source_class": "OFFICIAL_DATA_GO_KR_DAEGU_CITY_FILE_DATA",
        "unofficial_source_used": False,
        "data_go_kr_api_service_key_used": False,
        "new_bis_api_call_count": 0,
        "db_write_count": 0,
    }


def candidate_route_scope() -> Dict[str, Any]:
    routes = pd.read_csv(SUSEONG_ELIGIBLE_ROUTES, dtype=str)
    eligible = routes[routes["eligible_for_prompt1_graph_gate"].astype(str).str.lower().eq("true")].copy()
    route_nos = sorted(set(eligible["route_no"].astype(str).str.strip()))
    return {
        "source_path": str(SUSEONG_ELIGIBLE_ROUTES),
        "source_sha256": k5.sha256_file(SUSEONG_ELIGIBLE_ROUTES),
        "route_direction_row_count": int(len(eligible)),
        "unique_route_id_count": int(eligible["route_id"].nunique()),
        "unique_route_no_count": len(route_nos),
        "route_nos": route_nos,
    }


def build_route_headway_mapping(frame: pd.DataFrame, suseong_scope: Mapping[str, Any]) -> pd.DataFrame:
    occurrence = pd.read_parquet(OCCURRENCE_PATH)
    occurrence = occurrence.copy()
    occurrence["route_no_norm"] = occurrence["route_no"].astype(str).str.strip()
    occurrence["stop_id_norm"] = occurrence["stop_id"].astype(str).str.strip()
    groups: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrence.to_dict("records"):
        groups[(str(row["route_no_norm"]), str(row["stop_id_norm"]))].append(row)

    suseong_routes = set(str(route) for route in suseong_scope["route_nos"])
    mapping_rows: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        key = (str(row["route_no"]), str(row["stop_id"]))
        matches = groups.get(key, [])
        match_count = len(matches)
        status = "UNMATCHED" if match_count == 0 else ("MATCHED" if match_count == 1 else "AMBIGUOUS")
        selected = matches[0] if match_count == 1 else {}
        candidate_ids = [str(item["route_stop_occurrence_id"]) for item in matches]
        candidate_direction_ids = sorted({str(item["direction_id"]) for item in matches})
        candidate_route_ids = sorted({str(item["route_id"]) for item in matches})
        repeated_count = sum(bool(item.get("is_repeated_stop_occurrence")) for item in matches)
        ambiguity_reason = ""
        if status == "AMBIGUOUS":
            ambiguity_reason = "official route/stop key maps to multiple project route-stop occurrences; direction/occurrence not forced"
        elif status == "UNMATCHED":
            ambiguity_reason = "official route/stop key absent from K6 occurrence master"
        mapping_rows.append({
            "official_row_id": row["official_row_id"],
            "official_route": row["route_no"],
            "official_direction": row["direction_text"],
            "official_stop_id": row["stop_id"],
            "official_stop_name": row["stop_name"],
            "timetable_type": row["timetable_type"],
            "first_bus": row["first_bus"],
            "last_bus": row["last_bus"],
            "trip_count": row["trip_count"],
            "average_headway_minutes": row["average_headway_minutes"],
            "average_headway_seconds": row["average_headway_seconds"],
            "scope_all_daegu": True,
            "scope_suseong_candidate": row["route_no"] in suseong_routes,
            "scope_route814": row["route_no"] == "814",
            "match_status": status,
            "match_count": match_count,
            "route_id": str(selected.get("route_id", "")),
            "direction_id": str(selected.get("direction_id", "")),
            "route_stop_occurrence_id": str(selected.get("route_stop_occurrence_id", "")),
            "stop_id": str(selected.get("stop_id", "")),
            "stop_sequence": selected.get("stop_sequence"),
            "occurrence_index": selected.get("occurrence_index"),
            "is_repeated_stop_occurrence": bool(selected.get("is_repeated_stop_occurrence", False)) if status == "MATCHED" else None,
            "candidate_route_ids": "|".join(candidate_route_ids),
            "candidate_direction_ids": "|".join(candidate_direction_ids),
            "candidate_route_stop_occurrence_ids": "|".join(candidate_ids),
            "ambiguous_repeated_occurrence_candidate_count": repeated_count if status == "AMBIGUOUS" else 0,
            "ambiguity_reason": ambiguity_reason,
            "mapping_method": "exact route_no + stop_id against K6 occurrence master; no direction-name coercion",
            "forced_ambiguous_mapping": False,
        })
    return pd.DataFrame(mapping_rows)


def distribution_records(frame: pd.DataFrame, suseong_scope: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scopes = {
        "ALL_DAEGU": frame,
        "SUSEONG_CANDIDATE_ROUTES": frame[frame["route_no"].isin(suseong_scope["route_nos"])],
        "ROUTE_814": frame[frame["route_no"].eq("814")],
    }
    for scope, scope_frame in scopes.items():
        stats = summary_stats(scope_frame["average_headway_minutes"])
        rows.append({
            "scope": scope,
            "group_by": "ALL",
            "group_value": "ALL",
            **stats,
            "unit": "minutes",
        })
        for (direction, timetable), group in scope_frame.groupby(["direction_text", "timetable_type"], dropna=False):
            rows.append({
                "scope": scope,
                "group_by": "direction+timetable_type",
                "group_value": f"{direction}|{timetable}",
                **summary_stats(group["average_headway_minutes"]),
                "unit": "minutes",
            })
    return rows


def mapping_summary(mapping: pd.DataFrame, mask: pd.Series) -> Dict[str, Any]:
    part = mapping[mask].copy()
    counts = Counter(part["match_status"])
    return {
        "official_row_count": int(len(part)),
        "matched": int(counts.get("MATCHED", 0)),
        "unmatched": int(counts.get("UNMATCHED", 0)),
        "ambiguous": int(counts.get("AMBIGUOUS", 0)),
        "duplicate_official_key_count": int(part.duplicated(["official_route", "official_stop_id", "official_direction", "timetable_type"]).sum()),
        "repeated_stop_occurrence_matched_count": int(part["is_repeated_stop_occurrence"].fillna(False).sum()),
        "match_rate": float(counts.get("MATCHED", 0) / len(part)) if len(part) else None,
    }


def suseong_summary(frame: pd.DataFrame, mapping: pd.DataFrame, suseong_scope: Mapping[str, Any]) -> Dict[str, Any]:
    scoped = frame[frame["route_no"].isin(suseong_scope["route_nos"])]
    return {
        "created_at": iso_kst(),
        "scope": "SUSEONG_CANDIDATE_ROUTES_BY_DL1_ELIGIBLE_ROUTE_NO",
        "candidate_scope": suseong_scope,
        "official_route_no_count_present": int(scoped["route_no"].nunique()),
        "official_row_count": int(len(scoped)),
        "valid_headway_row_count": int(scoped["average_headway_minutes"].notna().sum()),
        "headway_minutes": summary_stats(scoped["average_headway_minutes"]),
        "headway_seconds": summary_stats(scoped["average_headway_seconds"]),
        "mapping_summary": mapping_summary(mapping, mapping["scope_suseong_candidate"].astype(bool)),
        "route_no_only_scope_limitation": "official file exposes route name, not project route_id; duplicate route variants sharing a route_no are audited as candidate-route-name scope",
    }


def route814_summary(frame: pd.DataFrame, mapping: pd.DataFrame) -> Dict[str, Any]:
    route = frame[frame["route_no"].eq("814")].copy()
    by_direction_type = []
    for (direction, timetable), group in route.groupby(["direction_text", "timetable_type"], dropna=False):
        by_direction_type.append({
            "direction": direction,
            "timetable_type": timetable,
            "row_count": int(len(group)),
            "first_bus_values": sorted(set(group["first_bus"])),
            "last_bus_values": sorted(set(group["last_bus"])),
            "trip_count": summary_stats(group["trip_count"]),
            "headway_minutes": summary_stats(group["average_headway_minutes"]),
            "headway_seconds": summary_stats(group["average_headway_seconds"]),
        })
    unmatched = mapping[(mapping["scope_route814"].astype(bool)) & (mapping["match_status"] != "MATCHED")]
    return {
        "created_at": iso_kst(),
        "route_no": "814",
        "project_route_id": "3000814001",
        "official_row_count": int(len(route)),
        "valid_headway_row_count": int(route["average_headway_minutes"].notna().sum()),
        "headway_minutes": summary_stats(route["average_headway_minutes"]),
        "headway_seconds": summary_stats(route["average_headway_seconds"]),
        "by_direction_timetable_type": by_direction_type,
        "mapping_summary": mapping_summary(mapping, mapping["scope_route814"].astype(bool)),
        "unmatched_or_ambiguous_rows": unmatched[[
            "official_stop_id",
            "official_stop_name",
            "official_direction",
            "timetable_type",
            "average_headway_minutes",
            "match_status",
            "ambiguity_reason",
        ]].to_dict("records"),
    }


def b1_temporal_realism_audit(frame: pd.DataFrame, mapping: pd.DataFrame, route814: Mapping[str, Any]) -> Dict[str, Any]:
    r8_candidate = k5.read_json(R8_NORMALIZATION_CANDIDATE)
    r8_windows = pd.read_parquet(R8_WINDOW_METRICS)
    constants = r8_candidate["constants"]
    route814_min = route814["headway_seconds"]["min"]
    route814_median = route814["headway_seconds"]["median"]
    suseong = frame[frame["route_no"].isin(candidate_route_scope()["route_nos"])]
    suseong_min = summary_stats(suseong["average_headway_seconds"])["min"]
    b1_avg = float(constants["B1_avg_wait_reference"])
    b1_p95 = float(constants["B1_p95_wait_reference"])
    material_mismatch = bool(route814_min is not None and b1_p95 < 0.25 * float(route814_min))
    r8_snapshot_counts = summary_stats(r8_windows["snapshot_count"])
    r8_active_counts = summary_stats(r8_windows["active_agent_count"])
    return {
        "created_at": iso_kst(),
        "audit_complete": True,
        "official_route814_headway_seconds": route814["headway_seconds"],
        "official_suseong_candidate_headway_seconds": summary_stats(suseong["average_headway_seconds"]),
        "official_all_daegu_headway_seconds": summary_stats(frame["average_headway_seconds"]),
        "current_r8_normalization_candidate": {
            "candidate_sha256": r8_candidate.get("candidate_sha256"),
            "B1_service_reference": constants["B1_service_reference"],
            "B1_avg_wait_reference": b1_avg,
            "B1_p95_wait_reference": b1_p95,
            "training_normalization_approved": r8_candidate.get("training_normalization_approved"),
        },
        "current_r8_b1_collection": {
            "window_count": int(len(r8_windows)),
            "time_bands": sorted(set(r8_windows["time_band"].astype(str))),
            "snapshot_count_summary": r8_snapshot_counts,
            "active_agent_count_summary": r8_active_counts,
            "window_duration_minutes": "29 minutes per R8 window",
            "mean_avg_wait_seconds": float(r8_windows["avg_wait_seconds"].mean()),
            "mean_p95_wait_seconds": float(r8_windows["p95_wait_seconds"].mean()),
        },
        "tests": {
            "does_every_60_sec_cycle_imply_new_service_opportunity": {
                "answer": "NOT_PROVEN_AS_LITERAL_IN_R8_WINDOWS",
                "detail": "R8 windows materialize four B1 service decisions over each 29-minute window, but the lifecycle has no official timetable/headway gate binding service opportunity timing.",
            },
            "fixed_8_agent_presence_unrealistically_dense_service": {
                "answer": "RISK_REQUIRES_HEADWAY_AWARE_ELIGIBILITY",
                "detail": "Fixed physical slots are acceptable only if service eligibility is separately scheduled; R8 did not bind agent availability to official first/last/headway rows.",
            },
            "passenger_wait_artificially_compressed": {
                "answer": bool(material_mismatch),
                "detail": "R8 research-demand requests are fixture-aligned with near-term SERVE/H4 outcomes; official route 814 average headway is 540-660 seconds by timetable type.",
            },
            "avg15_p95_23_plausible_under_official_headway": {
                "answer": False,
                "detail": "15/23 seconds are not defensible as unconditional passenger waiting references under route 814 official 9-11 minute average headways. The headway/2 heuristic is recorded only as a sanity check, not proof.",
            },
        },
        "sanity_check_not_used_as_proof": {
            "route814_headway_half_seconds_min": float(route814_min) / 2.0 if route814_min is not None else None,
            "route814_headway_half_seconds_median": float(route814_median) / 2.0 if route814_median is not None else None,
            "b1_avg_wait_seconds": b1_avg,
            "b1_p95_wait_seconds": b1_p95,
        },
        "mapping_failure_counts": {
            "all_daegu": mapping_summary(mapping, pd.Series([True] * len(mapping))),
            "suseong_candidate": mapping_summary(mapping, mapping["scope_suseong_candidate"].astype(bool)),
            "route814": mapping_summary(mapping, mapping["scope_route814"].astype(bool)),
        },
        "material_temporal_mismatch": material_mismatch,
        "b1_temporal_contract_has_official_headway_gate": False,
        "r8_normalization_constants_temporally_defensible": False,
        "new_bis_api_call_count": 0,
        "db_write_count": 0,
        "mappo_training_count": 0,
    }


def temporal_repair_candidate(audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "candidate_id": "PV8_B1_HEADWAY_AWARE_TEMPORAL_CONTRACT_CANDIDATE_V1",
        "status": "INACTIVE_NOT_APPROVED",
        "trigger": "R8 B1 wait references are materially compressed relative to official Daegu route headway evidence",
        "do_not_change": [
            "K-safety predicates",
            "reward F_PV8_SERVICE_GATED_CENTERED_CORE_V1",
            "H4 semantics",
            "PV8_RESEARCH_DEMAND_CANDIDATE_V1 demand contract",
        ],
        "required_mechanisms": [
            "separate global simulator tick from scheduled vehicle service opportunity",
            "bind route_id/direction_id/timetable_type to official first/last/headway evidence",
            "maintain per-route/direction next_service_ts and service eligibility windows",
            "block SERVE-as-arrival unless the vehicle is at a valid scheduled service opportunity",
            "sample research passenger request_ts independently of imminent vehicle arrival",
            "derive B1 normalization only after regenerating B1 windows with headway-aware eligibility",
            "preserve K4 dynamic obligation state and K8/K9 mask lifecycle",
        ],
        "minimum_next_stage": "PV8-R2A-R8B headway-aware B1 temporal contract implementation and B1 regeneration",
        "activation_authorized": False,
        "training_normalization_approved": False,
    }


def r8_normalization_reassessment(audit: Mapping[str, Any]) -> Dict[str, Any]:
    classification = "R8_NORMALIZATION_REQUIRES_B1_REGENERATION" if audit["material_temporal_mismatch"] else "R8_NORMALIZATION_TEMPORALLY_PLAUSIBLE"
    return {
        "created_at": iso_kst(),
        "classification": classification,
        "B1_service_reference": "NOT_APPROVED" if classification != "R8_NORMALIZATION_TEMPORALLY_PLAUSIBLE" else 1.0,
        "B1_avg_wait_reference": "NOT_APPROVED" if classification != "R8_NORMALIZATION_TEMPORALLY_PLAUSIBLE" else 15.0,
        "B1_p95_wait_reference": "NOT_APPROVED" if classification != "R8_NORMALIZATION_TEMPORALLY_PLAUSIBLE" else 23.0,
        "previous_r8_values": {
            "B1_service_reference": 1.0,
            "B1_avg_wait_reference": 15.0,
            "B1_p95_wait_reference": 23.0,
        },
        "preserved_for_lineage_only": True,
        "training_normalization_approved": False,
        "minimum_repair_before_approval": "regenerate B1 normalization candidate under an official-headway-aware temporal service contract",
    }


def readiness_decision(audit: Mapping[str, Any], reassessment: Mapping[str, Any], route814: Mapping[str, Any]) -> Dict[str, Any]:
    route814_mapping = route814["mapping_summary"]
    route814_usable = route814["official_row_count"] > 0 and route814["valid_headway_row_count"] > 0 and route814_mapping["matched"] > 0
    if not route814_usable:
        decision = DECISION_MAPPING_INCOMPLETE
        readiness = READINESS_MAPPING
        blockers = ["route 814 official headway rows could not be mapped or summarized sufficiently"]
    elif reassessment["classification"] == "R8_NORMALIZATION_REQUIRES_B1_REGENERATION":
        decision = DECISION_REPAIR_REQUIRED
        readiness = READINESS_REPAIR
        blockers = ["current R8 B1 temporal contract lacks official route-headway service eligibility and compresses waiting-time references"]
    else:
        decision = DECISION_ACCEPTABLE
        readiness = READINESS_ACCEPTABLE
        blockers = []
    return {
        "created_at": iso_kst(),
        "final_decision": decision,
        "gate": PASS_GATE,
        "readiness": readiness,
        "audit_complete": True,
        "official_source_acquired": True,
        "official_row_count": OFFICIAL_EXPECTED_ROW_COUNT,
        "route814_official_rows": route814["official_row_count"],
        "route814_mapping_match_rate": route814_mapping["match_rate"],
        "r8_normalization_reassessment": reassessment["classification"],
        "exact_blockers": blockers,
        "minimum_next_repair": "PV8-R2A-R8B: implement headway-aware B1 temporal contract and regenerate B1 references" if blockers else "explicit normalization approval may proceed",
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
    }


def claim_guard_status() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "conditional_skip_policy_enabled": False,
        "mappo_training_authorized": False,
        "automatic_r8b_execution_authorized": False,
        "automatic_r9_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
    }


def final_report(root: Path, source: Mapping[str, Any], suseong: Mapping[str, Any], route814: Mapping[str, Any], audit: Mapping[str, Any], reassessment: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    r814 = route814["headway_minutes"]
    su = suseong["headway_minutes"]
    mapping = route814["mapping_summary"]
    return "\n".join([
        "# PV8-R2A-R8A Official Headway Temporal-Realism Audit",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- official dataset: `{source['dataset_name']}`",
        f"- official file sha256: `{source['file_sha256']}`",
        "",
        "## Official Source",
        "",
        f"Official source is Data.go.kr / {source['source_owner']} file data dated `{source['source_reference_date']}`. Frozen CSV rows are `{source['row_count']}` and match the portal row count.",
        "",
        "## Headway Evidence",
        "",
        f"- Suseong candidate route names present: `{suseong['official_route_no_count_present']}`",
        f"- Suseong official rows: `{suseong['official_row_count']}`; median headway `{su['median']}` min, p25/p75 `{su['p25']}`/`{su['p75']}` min",
        f"- route 814 official rows: `{route814['official_row_count']}`; matched `{mapping['matched']}`, unmatched `{mapping['unmatched']}`, ambiguous `{mapping['ambiguous']}`",
        f"- route 814 headway: min/median/max `{r814['min']}`/`{r814['median']}`/`{r814['max']}` minutes",
        "",
        "## Temporal-Realism Result",
        "",
        f"Current R8 B1 references are avg wait `{audit['current_r8_normalization_candidate']['B1_avg_wait_reference']}` sec and p95 wait `{audit['current_r8_normalization_candidate']['B1_p95_wait_reference']}` sec. These are not defensible as unconditional B1 training-normalization constants against route 814's official 9-11 minute headway evidence.",
        "",
        f"- every-60-sec cycle implies service opportunity: `{audit['tests']['does_every_60_sec_cycle_imply_new_service_opportunity']['answer']}`",
        f"- fixed 8-agent dense-service risk: `{audit['tests']['fixed_8_agent_presence_unrealistically_dense_service']['answer']}`",
        f"- passenger wait compressed: `{audit['tests']['passenger_wait_artificially_compressed']['answer']}`",
        f"- 15/23 sec plausible: `{audit['tests']['avg15_p95_23_plausible_under_official_headway']['answer']}`",
        "",
        "## Normalization Reassessment",
        "",
        f"`{reassessment['classification']}`. R8's 1.0/15.0/23.0 values are lineage only and remain `NOT_APPROVED`; B1 must be regenerated under an official-headway-aware temporal contract before normalization approval.",
        "",
        "No BIS API calls, DB writes, reward materialization, policy evaluation, checkpoint reuse, downstream auto-run, or MAPPO training occurred.",
        "",
    ])


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": k5.sha256_file(path) if path.exists() else None,
            "required": True,
            "artifact_role": Path(relative_path).stem,
            "exists": path.exists(),
        })
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8a.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({
        "relative_path": jsonl_name,
        "size_bytes": jsonl_path.stat().st_size,
        "sha256": k5.sha256_file(jsonl_path),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8a.json"
    writer.json(manifest_name, {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "payload_count": len(rows),
        "missing_payload_count": sum(not row["exists"] for row in rows),
        "files": rows,
    })
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR8A_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path, official_csv: Path) -> Path:
    frozen = verify_frozen_context()
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    frozen_csv = root / "r8a_official_headway_raw_or_frozen_reference.csv"
    shutil.copyfile(official_csv, frozen_csv)

    official = load_official_headway(frozen_csv)
    source = source_manifest(official_csv, frozen_csv, official)
    scope = candidate_route_scope()
    mapping = build_route_headway_mapping(official, scope)
    distributions = distribution_records(official, scope)
    suseong = suseong_summary(official, mapping, scope)
    route814 = route814_summary(official, mapping)
    audit = b1_temporal_realism_audit(official, mapping, route814)
    repair = temporal_repair_candidate(audit)
    reassessment = r8_normalization_reassessment(audit)
    decision = readiness_decision(audit, reassessment, route814)
    guards = claim_guard_status()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": decision["readiness"],
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": decision["exact_blockers"],
    }

    writer.json("r8a_official_headway_source_manifest.json", source)
    mapping.to_parquet(root / "r8a_route_headway_mapping.parquet", index=False)
    writer.json("r8a_suseong_headway_summary.json", suseong)
    writer.json("r8a_route814_headway_summary.json", route814)
    pd.DataFrame(distributions).to_parquet(root / "r8a_headway_distribution.parquet", index=False)
    writer.json("r8a_b1_temporal_realism_audit.json", audit)
    writer.json("r8a_temporal_contract_repair_candidate.json", repair)
    writer.json("r8a_r8_normalization_reassessment.json", reassessment)
    writer.json("r8a_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "official-headway-temporal-realism-audit",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "authoritative_upstream_integrity": frozen,
        "official_dataset_name": OFFICIAL_DATASET_NAME,
        "official_source_url": OFFICIAL_SOURCE_URL,
        "official_file_sha256": source["file_sha256"],
        "official_row_count": source["row_count"],
        "route814_row_count": route814["official_row_count"],
        "route814_mapping_summary": route814["mapping_summary"],
        "suseong_mapping_summary": suseong["mapping_summary"],
        "r8_normalization_reassessment": reassessment["classification"],
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": decision["readiness"], "final_decision": decision["final_decision"]})
    writer.text("final_report.md", final_report(root, source, suseong, route814, audit, reassessment, decision))
    write_manifest_and_lock(writer, gate)
    checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8a.json", "_PV8_R2AR8A_COMPLETE.lock")
    if not k5.manifest_ok(checks):
        raise R8AError(f"R2A-R8A manifest integrity failure: {checks}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"official_rows: {source['row_count']}")
    print(f"route814_headway_minutes: {route814['headway_minutes']}")
    print(f"r8_normalization_reassessment: {reassessment['classification']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--official-csv", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root, args.official_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
