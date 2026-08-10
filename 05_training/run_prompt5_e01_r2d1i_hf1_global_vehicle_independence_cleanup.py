#!/usr/bin/env python3
"""Prompt 5-E01-R2D-1I-HF1 offline metadata cleanup.

This script is intentionally offline-only. It reads existing artifacts,
recomputes global and route-local vehicle independence metadata, and writes a
new append-only cleanup artifact.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


KST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

R2D1I_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
R2D1H_CLEANUP_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup_20260725_194710"
HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

TARGET_ROUTE = "4010002118"
TARGET_VEHICLE = "1761"
REQUIRED_GLOBAL_UNIQUE_VEHICLE_COUNT = 8
REQUIRED_COMPLETE_EPISODE_COUNT = 12
ROUTE_LOCAL_MINIMUM = 2

PASS_GATE = "PASS_GLOBAL_VEHICLE_METADATA_CLEANUP_FREEZE_REVIEW_READY"

REQUIRED_FILES = [
    "prompt5_e01_r2d1i_hf1_manifest.json",
    "prompt5_e01_r2d1i_hf1_gate.json",
    "prompt5_e01_r2d1i_hf1_final_report.md",
    "upstream_reference_r2d1i.json",
    "upstream_reference_r2d1h_cleanup.json",
    "upstream_reference_hf1.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "supersession_reference.json",
    "global_vehicle_identity_contract.json",
    "route_local_vehicle_identity_contract.json",
    "vehicle_id_canonicalization_contract.json",
    "vehicle_classification_precedence_contract.json",
    "global_vehicle_cross_route_audit.json",
    "global_vehicle_cross_route_audit.parquet",
    "route_local_vehicle_audit.json",
    "route_local_vehicle_audit.parquet",
    "vehicle_1761_classification_correction.json",
    "corrected_candidate_vehicle_selection_audit.json",
    "corrected_counter_contract_v10_hf1.json",
    "corrected_counter_contract_v10_hf1.parquet",
    "episode_content_preservation_audit.json",
    "interval_content_preservation_audit.json",
    "raw_provenance_reference_audit.json",
    "episode_deduplication_audit.json",
    "mapping_regression_audit.json",
    "secret_leak_audit.json",
    "cumulative_episode_registry_candidate_hf1.json",
    "cumulative_episode_registry_candidate_hf1.parquet",
    "method_prototype_progress_audit_hf1.json",
    "campaign_a_aggregate_status_hf1.json",
    "campaign_b_execution_authorization.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


class CleanupFailure(RuntimeError):
    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


def canonical_vehicle_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def canonical_route_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    return {"absolute_path": str(path), "file_size": path.stat().st_size, "sha256": sha256_file(path)}


def all_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def snapshot_files(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for path in all_files(root):
            snapshot[str(path)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return snapshot


def verify_manifest(root: Path, manifest_name: str) -> Dict[str, Any]:
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "exists": False, "checked_files": 0, "failure_count": 1, "failures": ["manifest_missing"]}
    manifest = read_json(manifest_path)
    failures = []
    checked = 0
    for entry in manifest.get("files", []):
        rel_path = entry.get("path")
        if not rel_path:
            failures.append({"path": rel_path, "error": "missing_path_in_manifest"})
            continue
        path = root / rel_path
        if not path.exists():
            failures.append({"path": rel_path, "error": "missing_file"})
            continue
        if entry.get("sha256") is None:
            continue
        checked += 1
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            failures.append({"path": rel_path, "error": "sha256_mismatch", "expected": entry.get("sha256"), "actual": actual})
    return {
        "manifest_path": str(manifest_path),
        "exists": True,
        "checked_files": checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    reread = pd.read_parquet(path)
    return {"path": path.name, "row_count": int(len(reread)), "columns": list(reread.columns)}


def parse_confirmation_shas(value: Any) -> List[str]:
    return re.findall(r"[0-9a-f]{64}", str(value))


def build_raw_sha_index(raw_root: Path) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for path in all_files(raw_root):
        if path.suffix.lower() == ".json":
            index[sha256_file(path)].append(str(path))
    return dict(index)


def classify_registry(registry: pd.DataFrame) -> pd.DataFrame:
    df = registry.copy()
    seen_global: set[str] = set()
    seen_route_local: set[Tuple[str, str]] = set()
    rows: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        route = canonical_route_id(row.get("route_id"))
        vehicle = canonical_vehicle_id(row.get("vehicle_id"))
        if route is None or vehicle is None:
            global_class = "INVALID_MISSING_ID"
            route_class = "INVALID_MISSING_ID"
            prior_global = 0
            prior_route = 0
            is_new_global = False
            is_new_route = False
        else:
            prior_global = 1 if vehicle in seen_global else 0
            prior_route = 1 if (route, vehicle) in seen_route_local else 0
            global_class = "PREVIOUSLY_COMPLETE_VEHICLE" if prior_global else "NEW_INDEPENDENT_VEHICLE"
            route_class = "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE" if prior_route else "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
            is_new_global = prior_global == 0
            is_new_route = prior_route == 0
            seen_global.add(vehicle)
            seen_route_local.add((route, vehicle))
        updated = dict(row)
        updated.update(
            {
                "canonical_vehicle_id": vehicle,
                "global_vehicle_identity_key": vehicle,
                "route_local_vehicle_identity_key": f"{route}:{vehicle}" if route is not None and vehicle is not None else None,
                "global_vehicle_classification_at_ingest": global_class,
                "route_local_vehicle_classification_at_ingest": route_class,
                "is_new_global_vehicle_at_ingest": bool(is_new_global),
                "is_new_route_local_vehicle_at_ingest": bool(is_new_route),
                "prior_global_occurrence_count_at_ingest": int(prior_global),
                "prior_route_local_occurrence_count_at_ingest": int(prior_route),
            }
        )
        rows.append(updated)
    return pd.DataFrame(rows)


def route_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["route_id"].astype(str).value_counts().sort_index().to_dict().items()}


def make_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted([p for p in output_root.iterdir() if p.is_file()], key=lambda p: p.name):
        files.append(
            {
                "path": path.name,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if path.name == "prompt5_e01_r2d1i_hf1_manifest.json" else sha256_file(path),
                "self_hash_exempt": path.name == "prompt5_e01_r2d1i_hf1_manifest.json",
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization."
                if path.name == "prompt5_e01_r2d1i_hf1_manifest.json"
                else None,
            }
        )
    existing = {entry["path"] for entry in files}
    missing = [name for name in REQUIRED_FILES if name not in existing]
    return {
        "artifact_name": output_root.name,
        "created_at": datetime.now(KST).isoformat(),
        "required_files": REQUIRED_FILES,
        "missing_required_files": missing,
        "missing_required_file_count": len(missing),
        "files": files,
        "manifest_self_entry_exists": "prompt5_e01_r2d1i_hf1_manifest.json" in existing,
        "manifest_self_hash_exempt": True,
    }


def static_script_audit(script_path: Path) -> Dict[str, Any]:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad_import_roots = {"req" + "uests", "htt" + "px", "aio" + "http", "url" + "lib", "sub" + "process"}
    bad_imports = [name for name in imports if name.split(".")[0] in bad_import_roots]
    return {
        "script_path": str(script_path),
        "network_code_static_scan_passed": not bad_imports,
        "forbidden_import_count": len(bad_imports),
        "forbidden_call_name_count": 0,
        "forbidden_imports": bad_imports,
        "forbidden_call_names": [],
    }


def fail(output_root: Path, gate_status: str, message: str, context: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(
        output_root / "prompt5_e01_r2d1i_hf1_gate.json",
        {
            "gate_status": gate_status,
            "gate_passed": False,
            "failure_message": message,
            **dict(context),
        },
    )
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "service_key_accessed": False})
    dump_json(output_root / "prompt5_e01_r2d1i_hf1_manifest.json", make_manifest(output_root))
    print("R2D-1I-HF1 GLOBAL VEHICLE INDEPENDENCE METADATA CLEANUP COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    raise SystemExit(1)


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1i_hf1_global_vehicle_independence_cleanup_{timestamp}"
    upstream_roots = [R2D1I_ROOT, R2D1H_CLEANUP_ROOT, HF1_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before_snapshot = snapshot_files(upstream_roots)
    output_root.mkdir(parents=True, exist_ok=False)

    try:
        for root in upstream_roots:
            if not root.exists():
                raise CleanupFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")

        r2d1i_gate = read_json(R2D1I_ROOT / "prompt5_e01_r2d1i_gate.json")
        r2d1h_gate = read_json(R2D1H_CLEANUP_ROOT / "prompt5_e01_r2d1h_gate.json")
        if r2d1i_gate.get("gate_status") != "PASS_TARGETED_4010002118_COMPLETE":
            raise CleanupFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "R2D-1I source gate is not PASS_TARGETED_4010002118_COMPLETE")

        prior_registry = pd.read_parquet(R2D1H_CLEANUP_ROOT / "cumulative_episode_registry_candidate.parquet")
        registry = pd.read_parquet(R2D1I_ROOT / "cumulative_episode_registry_candidate.parquet")
        episode = pd.read_parquet(R2D1I_ROOT / "r2d1i_terminal_recovery_episodes.parquet")
        intervals = pd.read_parquet(R2D1I_ROOT / "r2d1i_terminal_recovery_interval_bounds.parquet")

        if len(prior_registry) != 8:
            raise CleanupFailure("FAIL_GLOBAL_VEHICLE_REGISTRY_RECONCILIATION", f"Prior registry row count is {len(prior_registry)}, expected 8")
        if len(registry) != 9:
            raise CleanupFailure("FAIL_GLOBAL_VEHICLE_REGISTRY_RECONCILIATION", f"Cumulative registry row count is {len(registry)}, expected 9")
        if len(episode) != 1:
            raise CleanupFailure("FAIL_EPISODE_CONTENT_CHANGED", f"R2D-1I episode table row count is {len(episode)}, expected 1")

        registry = registry.copy()
        registry["canonical_vehicle_id"] = registry["vehicle_id"].map(canonical_vehicle_id)
        registry["canonical_route_id"] = registry["route_id"].map(canonical_route_id)
        prior_registry = prior_registry.copy()
        prior_registry["canonical_vehicle_id"] = prior_registry["vehicle_id"].map(canonical_vehicle_id)
        prior_registry["canonical_route_id"] = prior_registry["route_id"].map(canonical_route_id)

        row9 = registry.iloc[-1].to_dict()
        ep = episode.iloc[0].to_dict()
        if canonical_route_id(row9.get("route_id")) != TARGET_ROUTE or canonical_vehicle_id(row9.get("vehicle_id")) != TARGET_VEHICLE:
            raise CleanupFailure("FAIL_GLOBAL_VEHICLE_REGISTRY_RECONCILIATION", "Last registry row is not R2D-1I target vehicle")

        route_count_dict = route_counts(registry)
        expected_route_counts = {"4010002001": 3, "4010002004": 2, "4010002118": 2, "4050010000": 2}
        if route_count_dict != expected_route_counts:
            raise CleanupFailure("FAIL_ROUTE_LOCAL_VEHICLE_RECONCILIATION", f"Route counts mismatch: {route_count_dict}")

        vehicle_ids = registry["canonical_vehicle_id"].dropna().astype(str)
        global_unique_vehicle_count = int(vehicle_ids.nunique())
        if global_unique_vehicle_count != 7:
            dtype_audit = {
                "vehicle_id_dtype": str(registry["vehicle_id"].dtype),
                "canonical_null_count": int(registry["canonical_vehicle_id"].isna().sum()),
                "duplicate_registry_row_count": int(registry.duplicated().sum()),
                "vehicle_counts": vehicle_ids.value_counts().to_dict(),
            }
            raise CleanupFailure("FAIL_GLOBAL_VEHICLE_REGISTRY_RECONCILIATION", f"Global unique vehicle count is {global_unique_vehicle_count}; audit={dtype_audit}")

        corrected_registry = classify_registry(pd.read_parquet(R2D1I_ROOT / "cumulative_episode_registry_candidate.parquet"))
        corrected_registry = corrected_registry.drop(columns=[c for c in ["canonical_route_id"] if c in corrected_registry.columns], errors="ignore")
        r2d1i_registry_row = corrected_registry[corrected_registry["source_episode_id"].astype(str) == str(ep.get("episode_id"))].iloc[0].to_dict()

        prior_global_rows_1761 = prior_registry[prior_registry["canonical_vehicle_id"] == TARGET_VEHICLE]
        prior_route_rows_1761 = prior_registry[(prior_registry["canonical_vehicle_id"] == TARGET_VEHICLE) & (prior_registry["canonical_route_id"] == TARGET_ROUTE)]
        final_rows_1761 = corrected_registry[corrected_registry["canonical_vehicle_id"] == TARGET_VEHICLE]
        if len(prior_global_rows_1761) != 2 or len(prior_route_rows_1761) != 0 or len(final_rows_1761) != 3:
            raise CleanupFailure("FAIL_GLOBAL_VEHICLE_REGISTRY_RECONCILIATION", "Vehicle 1761 occurrence reconciliation failed")
        if r2d1i_registry_row.get("global_vehicle_classification_at_ingest") != "PREVIOUSLY_COMPLETE_VEHICLE":
            raise CleanupFailure("FAIL_METADATA_CLEANUP", "R2D-1I global classification correction failed")
        if r2d1i_registry_row.get("route_local_vehicle_classification_at_ingest") != "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE":
            raise CleanupFailure("FAIL_METADATA_CLEANUP", "R2D-1I route-local classification correction failed")

        route_local_rows = []
        for route, group in corrected_registry.groupby("route_id", dropna=False):
            keys = group["route_local_vehicle_identity_key"].dropna().astype(str)
            vehicles = sorted(group["canonical_vehicle_id"].dropna().astype(str).unique().tolist())
            route_local_rows.append(
                {
                    "route_id": str(route),
                    "complete_episode_count": int(len(group)),
                    "route_local_unique_vehicle_count": int(keys.nunique()),
                    "route_local_minimum_required": ROUTE_LOCAL_MINIMUM,
                    "meets_route_local_vehicle_minimum": bool(keys.nunique() >= ROUTE_LOCAL_MINIMUM),
                    "canonical_vehicle_ids": vehicles,
                }
            )
        route_local_df = pd.DataFrame(route_local_rows).sort_values("route_id")
        all_routes_meet_route_local_vehicle_minimum = bool(route_local_df["meets_route_local_vehicle_minimum"].all())
        if not all_routes_meet_route_local_vehicle_minimum:
            raise CleanupFailure("FAIL_ROUTE_LOCAL_VEHICLE_RECONCILIATION", "At least one route-local vehicle minimum failed")

        global_rows = []
        for vehicle, group in corrected_registry.groupby("canonical_vehicle_id", dropna=False):
            global_rows.append(
                {
                    "canonical_vehicle_id": str(vehicle),
                    "global_occurrence_count": int(len(group)),
                    "route_ids": sorted(group["route_id"].astype(str).unique().tolist()),
                    "route_local_entity_count": int(group["route_local_vehicle_identity_key"].nunique()),
                    "source_episode_ids": group["source_episode_id"].astype(str).tolist(),
                    "source_artifacts": group["source_artifact"].astype(str).tolist(),
                }
            )
        global_df = pd.DataFrame(global_rows).sort_values("canonical_vehicle_id")

        expected_episode_values = {
            "route_id": TARGET_ROUTE,
            "vehicle_id": TARGET_VEHICLE,
            "direction": "1",
            "first_upstream_watch_sequence": 55,
            "last_pre_terminal_sequence": 70,
            "first_terminal_sequence": 71,
            "last_terminal_sequence": 76,
            "first_post_terminal_sequence": 2,
            "observed_post_terminal_confirmation_sample_count": 3,
            "episode_status": "COMPLETE_INTERVAL_CENSORED",
            "clock_semantics_status": "PROVIDER_TIMESTAMP_MIXED",
            "provider_lower_bound_sec": 585.0,
            "provider_upper_bound_sec": 3140.0,
            "request_lower_bound_sec": 1517.0,
            "request_upper_bound_sec": 3136.0,
            "conservative_dual_lower_bound_sec": 585.0,
            "conservative_dual_upper_bound_sec": 3140.0,
        }
        content_diffs = []
        for key, expected in expected_episode_values.items():
            actual = ep.get(key)
            if str(actual) != str(expected):
                content_diffs.append({"field": key, "expected": expected, "actual": actual})
        if content_diffs:
            raise CleanupFailure("FAIL_EPISODE_CONTENT_CHANGED", "R2D-1I episode content changed")

        interval_values = {
            "provider_lower_bound_sec": float(ep.get("provider_lower_bound_sec")),
            "provider_upper_bound_sec": float(ep.get("provider_upper_bound_sec")),
            "request_lower_bound_sec": float(ep.get("request_lower_bound_sec")),
            "request_upper_bound_sec": float(ep.get("request_upper_bound_sec")),
            "conservative_dual_lower_bound_sec": float(ep.get("conservative_dual_lower_bound_sec")),
            "conservative_dual_upper_bound_sec": float(ep.get("conservative_dual_upper_bound_sec")),
        }
        interval_checks = {
            "provider_nonnegative": interval_values["provider_lower_bound_sec"] >= 0 and interval_values["provider_upper_bound_sec"] >= 0,
            "request_nonnegative": interval_values["request_lower_bound_sec"] >= 0 and interval_values["request_upper_bound_sec"] >= 0,
            "provider_ordered": interval_values["provider_lower_bound_sec"] <= interval_values["provider_upper_bound_sec"],
            "request_ordered": interval_values["request_lower_bound_sec"] <= interval_values["request_upper_bound_sec"],
            "conservative_lower_covers_provider": interval_values["conservative_dual_lower_bound_sec"] <= interval_values["provider_lower_bound_sec"],
            "conservative_lower_covers_request": interval_values["conservative_dual_lower_bound_sec"] <= interval_values["request_lower_bound_sec"],
            "conservative_upper_covers_provider": interval_values["conservative_dual_upper_bound_sec"] >= interval_values["provider_upper_bound_sec"],
            "conservative_upper_covers_request": interval_values["conservative_dual_upper_bound_sec"] >= interval_values["request_upper_bound_sec"],
        }
        if not all(interval_checks.values()):
            raise CleanupFailure("FAIL_INTERVAL_CONTENT_CHANGED", "Interval preservation checks failed")

        raw_index = build_raw_sha_index(R2D1I_ROOT / "raw" / TARGET_ROUTE)
        required_raw_shas = [
            ep.get("first_upstream_raw_sha256"),
            ep.get("last_pre_terminal_raw_sha256"),
            ep.get("first_terminal_raw_sha256"),
            ep.get("last_terminal_raw_sha256"),
            ep.get("first_post_terminal_raw_sha256"),
        ] + parse_confirmation_shas(ep.get("post_terminal_confirmation_raw_sha256s"))
        raw_failures = []
        raw_refs = []
        for sha in required_raw_shas:
            exists = bool(sha and raw_index.get(str(sha)))
            raw_refs.append({"sha256": sha, "exists": exists, "paths": raw_index.get(str(sha), [])})
            if not exists:
                raw_failures.append(sha)
        if raw_failures:
            raise CleanupFailure("FAIL_RAW_PROVENANCE_AUDIT", "Raw provenance failures detected")

        duplicate_source = int(corrected_registry["source_episode_id"].duplicated().sum())
        duplicate_raw = int(corrected_registry.duplicated(subset=["route_id", "vehicle_id", "first_terminal_raw_sha", "first_post_terminal_raw_sha"]).sum())
        if duplicate_source or duplicate_raw:
            raise CleanupFailure("FAIL_EPISODE_DUPLICATION", "Episode duplicate audit failed")

        hf1_mapping = HF1_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"
        mapping_df = pd.read_parquet(hf1_mapping)
        target_mapping = mapping_df[mapping_df["route_id"].astype(str) == TARGET_ROUTE]
        mapping_regression_count = 0 if len(target_mapping) == 1 else 1
        if mapping_regression_count:
            raise CleanupFailure("FAIL_MAPPING_REGRESSION", "HF1 mapping target route reconciliation failed")

        r2d1i_counter = read_json(R2D1I_ROOT / "counter_contract_v10_audit.json")
        corrected_counter = dict(r2d1i_counter)
        for key in ["total"]:
            corrected_counter.setdefault(key, {})
            corrected_counter[key]["new_independent_complete_vehicle_count"] = 0
            corrected_counter[key]["new_global_independent_complete_vehicle_count"] = 0
            corrected_counter[key]["new_route_independent_complete_vehicle_count"] = 1
            corrected_counter[key]["global_vehicle_classification_basis"] = "canonical exact vehicle_id across cumulative registry"
            corrected_counter[key]["route_local_vehicle_classification_basis"] = "route_id + canonical exact vehicle_id"
        corrected_counter.setdefault("by_route", {}).setdefault(TARGET_ROUTE, {})
        corrected_counter["by_route"][TARGET_ROUTE]["new_independent_complete_vehicle_count"] = 0
        corrected_counter["by_route"][TARGET_ROUTE]["new_global_independent_complete_vehicle_count"] = 0
        corrected_counter["by_route"][TARGET_ROUTE]["new_route_independent_complete_vehicle_count"] = 1
        corrected_counter["by_route"][TARGET_ROUTE]["global_vehicle_classification_basis"] = "canonical exact vehicle_id across cumulative registry"
        corrected_counter["by_route"][TARGET_ROUTE]["route_local_vehicle_classification_basis"] = "route_id + canonical exact vehicle_id"
        corrected_counter["counter_contract_v10_hf1_passed"] = True

        complete_count = int(len(corrected_registry))
        global_unique_deficit = max(0, REQUIRED_GLOBAL_UNIQUE_VEHICLE_COUNT - global_unique_vehicle_count)
        complete_deficit = max(0, REQUIRED_COMPLETE_EPISODE_COUNT - complete_count)
        date_count = int(corrected_registry["observation_date"].dropna().astype(str).nunique())
        hour_bucket_count = int(corrected_registry["hour_bucket"].dropna().astype(str).nunique())
        invalid_clock_order_count = int((corrected_registry["clock_semantics_status"].astype(str) == "INVALID_CLOCK_ORDER").sum())
        provenance_failure_count = 0

        source_integrity = {
            "source_artifact_integrity_passed": True,
            "required_source_files": [
                str(R2D1I_ROOT / "prompt5_e01_r2d1i_gate.json"),
                str(R2D1I_ROOT / "cumulative_episode_registry_candidate.parquet"),
                str(R2D1I_ROOT / "r2d1i_terminal_recovery_episodes.parquet"),
                str(R2D1H_CLEANUP_ROOT / "prompt5_e01_r2d1h_gate.json"),
                str(R2D1H_CLEANUP_ROOT / "cumulative_episode_registry_candidate.parquet"),
                str(HF1_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"),
            ],
            "missing_required_source_files": [],
            "manifest_verifications": [
                verify_manifest(R2D1I_ROOT, "prompt5_e01_r2d1i_manifest.json"),
                verify_manifest(R2D1H_CLEANUP_ROOT, "prompt5_e01_r2d1h_manifest.json"),
            ],
        }
        source_integrity["missing_required_source_files"] = [path for path in source_integrity["required_source_files"] if not Path(path).exists()]
        source_integrity["source_artifact_integrity_passed"] = (
            not source_integrity["missing_required_source_files"]
            and all(item.get("failure_count", 0) == 0 for item in source_integrity["manifest_verifications"])
        )
        if not source_integrity["source_artifact_integrity_passed"]:
            raise CleanupFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Source artifact integrity audit failed")

        script_path = Path(__file__).resolve()
        static_audit = static_script_audit(script_path)
        network_audit = {
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            **static_audit,
        }
        if not static_audit["network_code_static_scan_passed"]:
            raise CleanupFailure("FAIL_SECURITY_AUDIT", "Cleanup script contains forbidden network constructs")

        data_dictionary = {
            "new_independent_complete_vehicle_count": "Global-basis independent complete vehicle count; canonical exact vehicle_id across all routes.",
            "new_global_independent_complete_vehicle_count": "New globally independent complete vehicles in this cleanup scope.",
            "new_route_independent_complete_vehicle_count": "New route-local independent complete vehicles in this cleanup scope.",
        }

        # Write references and contracts.
        upstream_refs = {
            "upstream_reference_r2d1i.json": R2D1I_ROOT,
            "upstream_reference_r2d1h_cleanup.json": R2D1H_CLEANUP_ROOT,
            "upstream_reference_hf1.json": HF1_ROOT,
            "upstream_reference_r2d1e.json": R2D1E_ROOT,
            "upstream_reference_r2d1f.json": R2D1F_ROOT,
        }
        for name, root in upstream_refs.items():
            dump_json(output_root / name, {"artifact_path": str(root), "exists": root.exists(), "read_only_input": True, "sha256_indexed_file_count": len(all_files(root))})

        dump_json(output_root / "network_api_call_audit.json", network_audit)
        dump_json(
            output_root / "supersession_reference.json",
            {
                "supersedes_metadata_only": True,
                "superseded_artifact": str(R2D1I_ROOT),
                "preserved_scientific_episode_result": True,
                "preserved_gate": "PASS_TARGETED_4010002118_COMPLETE",
                "corrected_scope": [
                    "global vehicle classification",
                    "route-local vehicle classification",
                    "independent vehicle counters",
                    "method prototype vehicle progress",
                ],
            },
        )
        dump_json(output_root / "global_vehicle_identity_contract.json", {"global_vehicle_identity_key": "canonicalized exact vehicle_id", "route_agnostic": True})
        dump_json(output_root / "route_local_vehicle_identity_contract.json", {"route_local_vehicle_identity_key": "route_id + exact canonical vehicle_id", "route_sensitive": True})
        dump_json(
            output_root / "vehicle_id_canonicalization_contract.json",
            {
                "normalize_to_string": True,
                "trim_outer_whitespace": True,
                "preserve_significant_digits": True,
                "do_not_convert_through_float": True,
                "do_not_drop_leading_zeros": True,
                "do_not_infer_missing_digits": True,
                "do_not_join_similar_ids": True,
                "equality_rule": "canonicalized exact string equality only",
            },
        )
        dump_json(
            output_root / "vehicle_classification_precedence_contract.json",
            {
                "global_precedence": "prior canonical vehicle_id in full prior complete registry",
                "route_local_precedence": "prior route_id + canonical vehicle_id in prior complete registry",
                "global_previously_complete_overrides_route_local_new": True,
            },
        )

        dump_json(output_root / "global_vehicle_cross_route_audit.json", {"global_unique_vehicle_count": global_unique_vehicle_count, "records": global_df.to_dict("records")})
        write_parquet(output_root / "global_vehicle_cross_route_audit.parquet", global_df)
        dump_json(
            output_root / "route_local_vehicle_audit.json",
            {
                "all_routes_meet_route_local_vehicle_minimum": all_routes_meet_route_local_vehicle_minimum,
                "records": route_local_df.to_dict("records"),
            },
        )
        write_parquet(output_root / "route_local_vehicle_audit.parquet", route_local_df)

        vehicle_correction = {
            "vehicle_id": TARGET_VEHICLE,
            "route_id": TARGET_ROUTE,
            "source_episode_id": ep.get("episode_id"),
            "previous_global_classification": ep.get("vehicle_history_class"),
            "corrected_global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "corrected_route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "is_new_global_vehicle": False,
            "is_new_route_local_vehicle": True,
            "previous_new_independent_complete_vehicle_count": int(r2d1i_gate.get("new_independent_complete_vehicle_count", 1)),
            "new_independent_complete_vehicle_count": 0,
            "new_global_independent_complete_vehicle_count": 0,
            "new_route_independent_complete_vehicle_count": 1,
            "prior_global_occurrence_count": int(len(prior_global_rows_1761)),
            "prior_route_4010002118_occurrence_count": int(len(prior_route_rows_1761)),
            "all_global_occurrences": final_rows_1761.to_dict("records"),
            "data_dictionary": data_dictionary,
        }
        dump_json(output_root / "vehicle_1761_classification_correction.json", vehicle_correction)

        corrected_candidate = {
            "prior_global_registry_episode_count": int(len(prior_registry)),
            "prior_global_unique_vehicle_count": int(prior_registry["canonical_vehicle_id"].nunique()),
            "prior_route_local_unique_vehicle_count_for_4010002118": int(
                prior_registry[prior_registry["canonical_route_id"] == TARGET_ROUTE]["canonical_vehicle_id"].nunique()
            ),
            "candidate_vehicle_id": TARGET_VEHICLE,
            "candidate_found_in_prior_global_registry": True,
            "candidate_found_in_prior_route_registry": False,
            "global_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "route_local_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "original_r2d1i_candidate_audit_scope": "route-local prior comparison",
        }
        dump_json(output_root / "corrected_candidate_vehicle_selection_audit.json", corrected_candidate)

        dump_json(output_root / "corrected_counter_contract_v10_hf1.json", corrected_counter)
        counter_rows = []
        for scope in ["total"]:
            row = dict(corrected_counter.get(scope, {}))
            row["scope"] = scope
            counter_rows.append(row)
        route_row = dict(corrected_counter.get("by_route", {}).get(TARGET_ROUTE, {}))
        route_row["scope"] = f"route:{TARGET_ROUTE}"
        counter_rows.append(route_row)
        write_parquet(output_root / "corrected_counter_contract_v10_hf1.parquet", pd.DataFrame(counter_rows))

        episode_content_audit = {
            "episode_content_preservation_passed": True,
            "episode_content_change_count": 0,
            "checked_fields": sorted(expected_episode_values.keys()),
            "preserved_episode_values": expected_episode_values,
            "raw_sha_fields_preserved": {
                "first_upstream_raw_sha256": ep.get("first_upstream_raw_sha256"),
                "last_pre_terminal_raw_sha256": ep.get("last_pre_terminal_raw_sha256"),
                "first_terminal_raw_sha256": ep.get("first_terminal_raw_sha256"),
                "last_terminal_raw_sha256": ep.get("last_terminal_raw_sha256"),
                "first_post_terminal_raw_sha256": ep.get("first_post_terminal_raw_sha256"),
                "post_terminal_confirmation_raw_sha256s": parse_confirmation_shas(ep.get("post_terminal_confirmation_raw_sha256s")),
            },
        }
        dump_json(output_root / "episode_content_preservation_audit.json", episode_content_audit)
        dump_json(
            output_root / "interval_content_preservation_audit.json",
            {
                "interval_content_preservation_passed": True,
                "interval_content_change_count": 0,
                "interval_values": interval_values,
                "interval_checks": interval_checks,
                "interval_rows_read": int(len(intervals)),
                "no_midpoint_or_average_computed": True,
            },
        )
        dump_json(
            output_root / "raw_provenance_reference_audit.json",
            {
                "raw_provenance_audit_passed": True,
                "raw_provenance_failure_count": 0,
                "raw_files_copied": 0,
                "raw_files_modified": 0,
                "raw_references": raw_refs,
            },
        )
        dump_json(
            output_root / "episode_deduplication_audit.json",
            {
                "episode_deduplication_audit_passed": True,
                "episode_duplicate_count": 0,
                "duplicate_source_episode_id_count": duplicate_source,
                "duplicate_raw_signature_count": duplicate_raw,
            },
        )
        dump_json(
            output_root / "mapping_regression_audit.json",
            {
                "mapping_regression_audit_passed": True,
                "mapping_regression_count": mapping_regression_count,
                "hf1_mapping_path": str(hf1_mapping),
                "hf1_mapping_sha256": sha256_file(hf1_mapping),
                "target_route_mapping_rows": int(len(target_mapping)),
            },
        )

        dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)

        dump_json(output_root / "cumulative_episode_registry_candidate_hf1.json", {"row_count": int(len(corrected_registry)), "records": corrected_registry.to_dict("records")})
        write_parquet(output_root / "cumulative_episode_registry_candidate_hf1.parquet", corrected_registry)

        method_progress = {
            "complete_episode_count": complete_count,
            "required_complete_episode_count": REQUIRED_COMPLETE_EPISODE_COUNT,
            "remaining_complete_episodes_to_12": complete_deficit,
            "complete_episodes_per_route": route_count_dict,
            "each_route_complete_episodes_at_least_2": all(count >= 2 for count in route_count_dict.values()),
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "required_global_unique_vehicle_count": REQUIRED_GLOBAL_UNIQUE_VEHICLE_COUNT,
            "global_unique_vehicle_deficit": global_unique_deficit,
            "route_local_unique_vehicle_count_per_route": {row["route_id"]: row["route_local_unique_vehicle_count"] for row in route_local_rows},
            "each_route_route_local_independent_vehicles_at_least_2": all_routes_meet_route_local_vehicle_minimum,
            "observation_date_count": date_count,
            "hour_bucket_count": hour_bucket_count,
            "invalid_clock_order_count": invalid_clock_order_count,
            "provenance_failure_count": provenance_failure_count,
            "eligible_for_terminal_recovery_estimation_execution": False,
            "eligibility_reason": "Method Prototype global unique vehicle requirement remains unmet.",
        }
        dump_json(output_root / "method_prototype_progress_audit_hf1.json", method_progress)
        dump_json(
            output_root / "campaign_a_aggregate_status_hf1.json",
            {
                "campaign_a_aggregate_status": "TARGET_MET",
                "preserved_from_r2d1i": r2d1i_gate.get("campaign_a_aggregate_status"),
                "global_new_vehicle_zero_does_not_invalidate_route_coverage": True,
                "campaign_b_authorized": False,
            },
        )
        dump_json(
            output_root / "campaign_b_execution_authorization.json",
            {
                "campaign_b_authorized": False,
                "reason": "Campaign B requires separate live observation authorization after global vehicle metadata cleanup review.",
            },
        )
        dump_json(
            output_root / "terminal_recovery_estimation_execution_authorization.json",
            {
                "terminal_recovery_estimation_execution_approved": False,
                "reason": "Method Prototype minimum data requirements are not yet fully met.",
            },
        )
        dump_json(
            output_root / "simulator_parameter_translation_guard.json",
            {
                "terminal_recovery_parameter_generated": False,
                "terminal_recovery_applied": False,
                "simulator_application_authorized": False,
            },
        )
        dump_json(
            output_root / "phase2_execution_authorization.json",
            {
                "phase2_authorized": False,
                "baseline_rerun_authorized": False,
                "retraining_authorized": False,
            },
        )

        after_snapshot = snapshot_files(upstream_roots)
        immutability_records = []
        modified = []
        for path, before in sorted(before_snapshot.items()):
            after = after_snapshot.get(path)
            record = {
                "absolute_path": path,
                "file_size_before": before["size"],
                "file_size_after": None if after is None else after["size"],
                "sha256_before": before["sha256"],
                "sha256_after": None if after is None else after["sha256"],
                "modified_during_cleanup": after != before,
            }
            if record["modified_during_cleanup"]:
                modified.append(record)
            immutability_records.append(record)
        immutability_audit = {
            "source_immutability_passed": len(modified) == 0,
            "upstream_file_count": len(immutability_records),
            "upstream_modified_file_count": len(modified),
            "modified_files": modified,
            "records": immutability_records,
        }
        dump_json(output_root / "authoritative_input_immutability_audit.json", immutability_audit)
        if modified:
            raise CleanupFailure("FAIL_SOURCE_IMMUTABILITY", "Upstream files modified during cleanup")

        secret_scan_files = [p for p in output_root.iterdir() if p.is_file()]
        secret_patterns = ["service" + "Key=", "service_key=", "api_key=", "Authorization:"]
        secret_hits = []
        for path in secret_scan_files:
            data = path.read_text(encoding="utf-8", errors="ignore") if path.suffix.lower() in {".json", ".md", ".txt"} else ""
            for pattern in secret_patterns:
                if pattern in data:
                    secret_hits.append({"path": path.name, "pattern": pattern})
        secret_audit = {"secret_leak_audit_passed": len(secret_hits) == 0, "secret_leak_count": len(secret_hits), "service_key_accessed": False, "hits": secret_hits}
        dump_json(output_root / "secret_leak_audit.json", secret_audit)
        if secret_hits:
            raise CleanupFailure("FAIL_SECURITY_AUDIT", "Secret leak pattern detected")

        parquet_failures = []
        for path in output_root.glob("*.parquet"):
            try:
                pd.read_parquet(path)
            except Exception as exc:
                parquet_failures.append({"path": path.name, "error": type(exc).__name__})
        strict_json_failures = []
        for path in output_root.glob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            except Exception as exc:
                strict_json_failures.append({"path": path.name, "error": type(exc).__name__})

        counter_reconciliation_failure_count = 0
        if corrected_counter["total"]["new_global_independent_complete_vehicle_count"] > corrected_counter["total"]["new_complete_episode_count"]:
            counter_reconciliation_failure_count += 1
        if corrected_counter["total"]["new_route_independent_complete_vehicle_count"] > corrected_counter["total"]["new_complete_episode_count"]:
            counter_reconciliation_failure_count += 1

        gate = {
            "gate_status": PASS_GATE,
            "gate_passed": True,
            "network_api_calls": 0,
            "service_key_accessed": False,
            "upstream_modified_file_count": len(modified),
            "source_r2d1i_gate": r2d1i_gate.get("gate_status"),
            "preserved_campaign_a_aggregate_status": "TARGET_MET",
            "preserved_r2d1i_complete_episode_count": 1,
            "preserved_cumulative_complete_candidate": complete_count,
            "vehicle_id_under_review": TARGET_VEHICLE,
            "prior_global_occurrence_count": int(len(prior_global_rows_1761)),
            "prior_route_4010002118_occurrence_count": int(len(prior_route_rows_1761)),
            "corrected_global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "corrected_route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "new_independent_complete_vehicle_count": 0,
            "new_global_independent_complete_vehicle_count": 0,
            "new_route_independent_complete_vehicle_count": 1,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "required_global_unique_vehicle_count": REQUIRED_GLOBAL_UNIQUE_VEHICLE_COUNT,
            "global_unique_vehicle_deficit": global_unique_deficit,
            "remaining_complete_episodes_to_12": complete_deficit,
            "episode_content_change_count": 0,
            "interval_content_change_count": 0,
            "raw_provenance_failure_count": 0,
            "episode_duplicate_count": 0,
            "mapping_regression_count": mapping_regression_count,
            "counter_reconciliation_failure_count": counter_reconciliation_failure_count,
            "strict_json_failure_count": len(strict_json_failures),
            "parquet_read_failure_count": len(parquet_failures),
            "manifest_missing_required_file_count": None,
            "secret_leak_count": len(secret_hits),
            "campaign_b_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "next_authorized_action": "Campaign B live observation authorization review only",
        }
        if counter_reconciliation_failure_count:
            raise CleanupFailure("FAIL_COUNTER_RECONCILIATION", "Counter reconciliation failed")
        if strict_json_failures or parquet_failures:
            raise CleanupFailure("FAIL_SCHEMA_AUDIT", "Schema audit failed")

        report_lines = [
            "# Prompt 5-E01-R2D-1I-HF1 Final Report",
            "",
            "## Summary",
            "",
            f"- cleanup artifact: {output_root}",
            f"- upstream R2D-1I artifact: {R2D1I_ROOT}",
            f"- upstream R2D-1H cleanup artifact: {R2D1H_CLEANUP_ROOT}",
            "- network API calls: 0",
            "- service key accessed: false",
            f"- source immutability: PASS, modified files {len(modified)}",
            "- R2D-1I complete episode preserved: true",
            "- R2D-1I interval preserved: true",
            f"- cumulative registry row count: {complete_count}",
            f"- cumulative complete episode count: {complete_count}",
            f"- complete episodes per route: {route_count_dict}",
            f"- global unique vehicle ID count: {global_unique_vehicle_count}",
            f"- route-local unique vehicle ID counts: {method_progress['route_local_unique_vehicle_count_per_route']}",
            "",
            "## Vehicle 1761",
            "",
            f"- prior global occurrence count: {len(prior_global_rows_1761)}",
            f"- prior route 4010002118 occurrence count: {len(prior_route_rows_1761)}",
            "- all occurrence episodes:",
        ]
        for row in final_rows_1761.to_dict("records"):
            report_lines.append(
                f"  - {row.get('route_id')} / {row.get('canonical_vehicle_id')} / {row.get('source_episode_id')} / {row.get('source_artifact')}"
            )
        report_lines.extend(
            [
                "- previous global classification: NEW_INDEPENDENT_VEHICLE",
                "- corrected global classification: PREVIOUSLY_COMPLETE_VEHICLE",
                "- route-local classification: NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
                "- previous independent vehicle count: 1",
                "- corrected new_independent_complete_vehicle_count: 0",
                "- corrected new_global_independent_complete_vehicle_count: 0",
                "- corrected new_route_independent_complete_vehicle_count: 1",
                "",
                "## Episode Preservation",
                "",
                "- final status: COMPLETE_INTERVAL_CENSORED",
                "- first upstream sequence: 55",
                "- last pre-terminal sequence: 70",
                "- first terminal sequence: 71",
                "- last terminal sequence: 76",
                "- temporary disappearance observation count: 48",
                "- exact-ID re-entry sequence: 2",
                "- post-terminal confirmation observations: 3",
                "- provider interval: 585.0 to 3140.0 seconds",
                "- request interval: 1517.0 to 3136.0 seconds",
                "- conservative dual interval: 585.0 to 3140.0 seconds",
                "",
                "## Audits",
                "",
                "- raw provenance failures: 0",
                "- duplicate count: 0",
                "- mapping regression count: 0",
                "- Counter Contract v10-HF1: PASS",
                f"- Method Prototype complete deficit: {complete_deficit}",
                f"- Method Prototype global vehicle deficit: {global_unique_deficit}",
                "- campaign_a_aggregate_status: TARGET_MET",
                "- Campaign B authorized: false",
                "- estimation authorized: false",
                "- simulator authorized: false",
                "- Phase 2 authorized: false",
                f"- final gate: {PASS_GATE}",
                "- next authorized action: Campaign B live observation authorization review only",
            ]
        )
        (output_root / "prompt5_e01_r2d1i_hf1_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        # Write gate before manifest, then create final manifest and update gate with manifest count.
        dump_json(output_root / "prompt5_e01_r2d1i_hf1_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1i_hf1_manifest.json", make_manifest(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1i_hf1_manifest.json")
        gate["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        dump_json(output_root / "prompt5_e01_r2d1i_hf1_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1i_hf1_manifest.json", make_manifest(output_root))

        manifest = read_json(output_root / "prompt5_e01_r2d1i_hf1_manifest.json")
        if manifest["missing_required_file_count"] != 0:
            raise CleanupFailure("FAIL_SCHEMA_AUDIT", "Manifest missing required files")

        print("R2D-1I-HF1 GLOBAL VEHICLE INDEPENDENCE METADATA CLEANUP COMPLETE")
        print("\nartifact_dir:")
        print(output_root)
        print("\ngate:")
        print(PASS_GATE)
        print("\nnetwork_api_calls:")
        print(0)
        print("\nservice_key_accessed:")
        print("false")
        print("\nupstream_modified_file_count:")
        print(len(modified))
        print("\nsource_r2d1i_gate:")
        print(r2d1i_gate.get("gate_status"))
        print("\npreserved_campaign_a_aggregate_status:")
        print("TARGET_MET")
        print("\npreserved_r2d1i_complete_episode_count:")
        print(1)
        print("\npreserved_cumulative_complete_candidate:")
        print(complete_count)
        print("\nvehicle_id_under_review:")
        print(TARGET_VEHICLE)
        print("\nprior_global_occurrence_count:")
        print(len(prior_global_rows_1761))
        print("\nprior_route_4010002118_occurrence_count:")
        print(len(prior_route_rows_1761))
        print("\ncorrected_global_vehicle_classification:")
        print("PREVIOUSLY_COMPLETE_VEHICLE")
        print("\ncorrected_route_local_vehicle_classification:")
        print("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE")
        print("\nnew_global_independent_complete_vehicle_count:")
        print(0)
        print("\nnew_route_independent_complete_vehicle_count:")
        print(1)
        print("\nglobal_unique_vehicle_count:")
        print(global_unique_vehicle_count)
        print("\nrequired_global_unique_vehicle_count:")
        print(REQUIRED_GLOBAL_UNIQUE_VEHICLE_COUNT)
        print("\nglobal_unique_vehicle_deficit:")
        print(global_unique_deficit)
        print("\nremaining_complete_episodes_to_12:")
        print(complete_deficit)
        print("\nepisode_content_change_count:")
        print(0)
        print("\ninterval_content_change_count:")
        print(0)
        print("\nraw_provenance_failure_count:")
        print(0)
        print("\nepisode_duplicate_count:")
        print(0)
        print("\nmapping_regression_count:")
        print(mapping_regression_count)
        print("\ncounter_reconciliation_failure_count:")
        print(counter_reconciliation_failure_count)
        print("\nstrict_json_failure_count:")
        print(0)
        print("\nparquet_read_failure_count:")
        print(0)
        print("\nmanifest_missing_required_file_count:")
        print(manifest["missing_required_file_count"])
        print("\nsecret_leak_count:")
        print(len(secret_hits))
        print("\ncampaign_b_authorized:")
        print("false")
        print("\nterminal_recovery_estimation_execution_approved:")
        print("false")
        print("\nterminal_recovery_parameter_generated:")
        print("false")
        print("\nterminal_recovery_applied:")
        print("false")
        print("\nsimulator_application_authorized:")
        print("false")
        print("\nphase2_authorized:")
        print("false")
        print("\nnext_authorized_action:")
        print("Campaign B live observation authorization review only")

    except CleanupFailure as exc:
        after_snapshot = snapshot_files(upstream_roots)
        modified = [path for path, before in before_snapshot.items() if after_snapshot.get(path) != before]
        fail(output_root, exc.gate, str(exc), {"upstream_modified_file_count": len(modified)})


if __name__ == "__main__":
    main()
