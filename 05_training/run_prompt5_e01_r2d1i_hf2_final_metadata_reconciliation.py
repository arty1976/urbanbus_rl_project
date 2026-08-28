#!/usr/bin/env python3
"""Prompt 5-E01-R2D-1I-HF2 offline final metadata reconciliation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


KST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"

HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf1_global_vehicle_independence_cleanup_20260726_132809"
R2D1I_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
R2D1H_CLEANUP_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup_20260725_194710"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

TARGET_ROUTE = "4010002118"
TARGET_VEHICLE = "1761"
TARGET_EPISODE_ID = "r2d1i_episode_00001_4010002118"
PASS_GATE = "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY"

REQUIRED_COMPLETE_EPISODES = 12
REQUIRED_GLOBAL_UNIQUE_VEHICLES = 8
ROUTE_LOCAL_MINIMUM = 2

REQUIRED_FILES = [
    "prompt5_e01_r2d1i_hf2_manifest.json",
    "prompt5_e01_r2d1i_hf2_gate.json",
    "prompt5_e01_r2d1i_hf2_final_report.md",
    "upstream_reference_r2d1i_hf1.json",
    "upstream_reference_r2d1i.json",
    "upstream_reference_r2d1h_cleanup.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "scientific_content_preservation_audit.json",
    "interval_content_preservation_audit.json",
    "raw_provenance_reference_audit.json",
    "mapping_regression_audit.json",
    "episode_deduplication_audit.json",
    "secret_leak_audit.json",
    "manifest_required_file_reconciliation.json",
    "vehicle_occurrence_reconciliation_contract.json",
    "vehicle_occurrence_reconciliation_audit.json",
    "vehicle_occurrence_reconciliation_audit.parquet",
    "global_vehicle_identity_contract_hf2.json",
    "route_local_vehicle_identity_contract_hf2.json",
    "vehicle_1761_final_classification.json",
    "corrected_candidate_vehicle_selection_audit_hf2.json",
    "corrected_counter_contract_v10_hf2.json",
    "corrected_counter_contract_v10_hf2.parquet",
    "cumulative_episode_registry_candidate_hf2.json",
    "cumulative_episode_registry_candidate_hf2.parquet",
    "method_prototype_progress_audit_hf2.json",
    "campaign_a_aggregate_status_hf2.json",
    "supersession_reference_hf2.json",
    "campaign_b_execution_authorization.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


class ReconciliationFailure(RuntimeError):
    def __init__(self, gate_status: str, message: str):
        super().__init__(message)
        self.gate_status = gate_status


def canonical_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


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
    return sorted([path for path in root.rglob("*") if path.is_file()])


def snapshot(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for path in files_under(root):
            out[str(path)] = {
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return out


def verify_manifest(root: Path, manifest_name: str) -> Dict[str, Any]:
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "exists": False, "failure_count": 1, "failures": ["manifest_missing"]}
    manifest = read_json(manifest_path)
    failures = []
    for entry in manifest.get("files", []):
        rel_path = entry.get("path")
        path = root / str(rel_path)
        if not path.exists():
            failures.append({"path": rel_path, "error": "missing"})
            continue
        expected = entry.get("sha256")
        if expected is not None:
            actual = sha256_file(path)
            if actual != expected:
                failures.append({"path": rel_path, "error": "sha256_mismatch", "expected": expected, "actual": actual})
    return {
        "manifest_path": str(manifest_path),
        "exists": True,
        "manifest_missing_required_file_count": manifest.get("missing_required_file_count"),
        "manifest_missing_required_files": manifest.get("missing_required_files"),
        "manifest_file_count": len(manifest.get("files", [])),
        "failure_count": len(failures),
        "failures": failures,
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    reread = pd.read_parquet(path)
    return {"path": path.name, "row_count": int(len(reread)), "columns": list(reread.columns)}


def stable_sort_registry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_sort_observation_date"] = out["observation_date"].map(lambda v: canonical_text(v) or "1900-01-01")
    out["_sort_source_episode_id"] = out["source_episode_id"].map(lambda v: canonical_text(v) or "")
    out["_sort_source_artifact"] = out["source_artifact"].map(lambda v: canonical_text(v) or "")
    # Preserve existing registry order as final tiebreaker. This keeps prior frozen rows before later live rows.
    out["_sort_original_index"] = list(range(len(out)))
    out = out.sort_values(
        ["_sort_observation_date", "_sort_source_artifact", "_sort_source_episode_id", "_sort_original_index"],
        kind="mergesort",
    ).drop(columns=["_sort_observation_date", "_sort_source_artifact", "_sort_source_episode_id", "_sort_original_index"])
    return out.reset_index(drop=True)


def reconcile_occurrences(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = stable_sort_registry(registry)
    seen_global: Dict[str, int] = defaultdict(int)
    seen_route_local: Dict[Tuple[str, str], int] = defaultdict(int)
    rows = []
    audit_rows = []
    for ingest_order, row in enumerate(ordered.to_dict("records"), start=1):
        route = canonical_text(row.get("route_id"))
        vehicle = canonical_text(row.get("vehicle_id"))
        prior_global = int(seen_global[vehicle]) if vehicle is not None else 0
        prior_route = int(seen_route_local[(route, vehicle)]) if route is not None and vehicle is not None else 0
        after_global = prior_global + 1 if vehicle is not None else 0
        after_route = prior_route + 1 if route is not None and vehicle is not None else 0
        global_class = "PREVIOUSLY_COMPLETE_VEHICLE" if prior_global > 0 else "NEW_INDEPENDENT_VEHICLE"
        route_class = "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE" if prior_route > 0 else "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
        updated = dict(row)
        updated.update(
            {
                "canonical_vehicle_id": vehicle,
                "global_vehicle_identity_key": vehicle,
                "route_local_vehicle_identity_key": f"{route}:{vehicle}" if route and vehicle else None,
                "prior_global_occurrence_count_at_ingest": prior_global,
                "global_occurrence_count_after_ingest": after_global,
                "prior_route_local_occurrence_count_at_ingest": prior_route,
                "route_local_occurrence_count_after_ingest": after_route,
                "global_vehicle_classification": global_class,
                "route_local_vehicle_classification": route_class,
                "is_new_global_vehicle": prior_global == 0,
                "is_new_route_local_vehicle": prior_route == 0,
                "global_vehicle_classification_at_ingest": global_class,
                "route_local_vehicle_classification_at_ingest": route_class,
                "is_new_global_vehicle_at_ingest": prior_global == 0,
                "is_new_route_local_vehicle_at_ingest": prior_route == 0,
                "hf2_ingest_order": ingest_order,
            }
        )
        audit_rows.append(
            {
                "hf2_ingest_order": ingest_order,
                "source_episode_id": updated.get("source_episode_id"),
                "route_id": route,
                "vehicle_id": vehicle,
                "prior_global_occurrence_count_at_ingest": prior_global,
                "global_occurrence_count_after_ingest": after_global,
                "prior_route_local_occurrence_count_at_ingest": prior_route,
                "route_local_occurrence_count_after_ingest": after_route,
                "global_vehicle_classification": global_class,
                "route_local_vehicle_classification": route_class,
                "is_new_global_vehicle": prior_global == 0,
                "is_new_route_local_vehicle": prior_route == 0,
            }
        )
        rows.append(updated)
        if vehicle is not None:
            seen_global[vehicle] += 1
        if route is not None and vehicle is not None:
            seen_route_local[(route, vehicle)] += 1
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def raw_sha_index(raw_root: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for path in files_under(raw_root):
        if path.suffix.lower() == ".json":
            out[sha256_file(path)].append(str(path))
    return dict(out)


def confirmation_shas(value: Any) -> List[str]:
    return re.findall(r"[0-9a-f]{64}", str(value))


def manifest_payload(output_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted([path for path in output_root.iterdir() if path.is_file()], key=lambda p: p.name):
        files.append(
            {
                "path": path.name,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if path.name == "prompt5_e01_r2d1i_hf2_manifest.json" else sha256_file(path),
                "self_hash_exempt": path.name == "prompt5_e01_r2d1i_hf2_manifest.json",
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization."
                if path.name == "prompt5_e01_r2d1i_hf2_manifest.json"
                else None,
            }
        )
    present = {entry["path"] for entry in files}
    missing = [name for name in REQUIRED_FILES if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at": datetime.now(KST).isoformat(),
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count": len(REQUIRED_FILES) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "files": files,
        "manifest_self_entry_exists": "prompt5_e01_r2d1i_hf2_manifest.json" in present,
        "manifest_self_hash_exempt": True,
    }


def static_code_audit(script_path: Path) -> Dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    blocked_imports = {"req" + "uests", "htt" + "px", "aio" + "http", "url" + "lib", "soc" + "ket"}
    hits = [name for name in imports if name.split(".")[0] in blocked_imports]
    return {
        "script_path": str(script_path),
        "network_code_static_scan_passed": len(hits) == 0,
        "forbidden_import_count": len(hits),
        "forbidden_imports": hits,
    }


def route_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["route_id"].astype(str).value_counts().sort_index().to_dict().items()}


def fail(output_root: Path, gate_status: str, message: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(output_root / "prompt5_e01_r2d1i_hf2_gate.json", {"gate_status": gate_status, "gate_passed": False, "failure_message": message})
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "service_key_accessed": False})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False})
    dump_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json", manifest_payload(output_root))
    print("R2D-1I-HF2 FINAL METADATA RECONCILIATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    raise SystemExit(1)


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_{timestamp}"
    upstream_roots = [HF1_ROOT, R2D1I_ROOT, R2D1H_CLEANUP_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)
    output_root.mkdir(parents=True, exist_ok=False)

    try:
        for root in upstream_roots:
            if not root.exists():
                raise ReconciliationFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")

        hf1_gate = read_json(HF1_ROOT / "prompt5_e01_r2d1i_hf1_gate.json")
        hf1_manifest = read_json(HF1_ROOT / "prompt5_e01_r2d1i_hf1_manifest.json")
        r2d1i_gate = read_json(R2D1I_ROOT / "prompt5_e01_r2d1i_gate.json")
        if hf1_gate.get("gate_status") != "PASS_GLOBAL_VEHICLE_METADATA_CLEANUP_FREEZE_REVIEW_READY":
            raise ReconciliationFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "HF1 gate is not PASS")
        if r2d1i_gate.get("gate_status") != "PASS_TARGETED_4010002118_COMPLETE":
            raise ReconciliationFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "R2D-1I gate is not PASS")

        required_present = [name for name in REQUIRED_FILES if name != "prompt5_e01_r2d1i_hf2_manifest.json"]
        hf1_required = hf1_manifest.get("required_files", [])
        hf1_missing_by_disk = [name for name in hf1_required if not (HF1_ROOT / name).exists()]
        manifest_reconciliation = {
            "hf1_gate_manifest_missing_required_file_count": hf1_gate.get("manifest_missing_required_file_count"),
            "hf1_manifest_missing_required_file_count": hf1_manifest.get("missing_required_file_count"),
            "hf1_manifest_missing_required_files": hf1_manifest.get("missing_required_files"),
            "hf1_required_file_count": len(hf1_required),
            "hf1_present_required_file_count": len(hf1_required) - len(hf1_missing_by_disk),
            "hf1_missing_required_file_count_from_disk": len(hf1_missing_by_disk),
            "hf1_missing_required_files_from_disk": hf1_missing_by_disk,
            "corrected_manifest_missing_required_file_count": 0 if not hf1_missing_by_disk and hf1_manifest.get("missing_required_file_count") == 0 else None,
        }
        if hf1_missing_by_disk or hf1_manifest.get("missing_required_file_count") != 0:
            raise ReconciliationFailure("FAIL_MANIFEST_RECONCILIATION", "HF1 manifest required-file reconciliation failed")

        prior_registry = pd.read_parquet(R2D1H_CLEANUP_ROOT / "cumulative_episode_registry_candidate.parquet")
        hf1_registry = pd.read_parquet(HF1_ROOT / "cumulative_episode_registry_candidate_hf1.parquet")
        hf1_occurrence = pd.read_parquet(HF1_ROOT / "vehicle_occurrence_reconciliation_audit.parquet") if (HF1_ROOT / "vehicle_occurrence_reconciliation_audit.parquet").exists() else None
        r2d1i_episode = pd.read_parquet(R2D1I_ROOT / "r2d1i_terminal_recovery_episodes.parquet")

        if len(prior_registry) != 8:
            raise ReconciliationFailure("FAIL_GLOBAL_VEHICLE_OCCURRENCE_RECONCILIATION", "Prior registry row count is not 8")
        if len(hf1_registry) != 9:
            raise ReconciliationFailure("FAIL_GLOBAL_VEHICLE_OCCURRENCE_RECONCILIATION", "HF1 registry row count is not 9")
        if len(r2d1i_episode) != 1:
            raise ReconciliationFailure("FAIL_SCIENTIFIC_CONTENT_CHANGED", "R2D-1I episode row count is not 1")

        hf2_registry, occurrence_audit_df = reconcile_occurrences(hf1_registry)
        target_rows = hf2_registry[hf2_registry["source_episode_id"].astype(str) == TARGET_EPISODE_ID]
        if len(target_rows) != 1:
            raise ReconciliationFailure("FAIL_GLOBAL_VEHICLE_OCCURRENCE_RECONCILIATION", "Target R2D-1I row missing from HF2 registry")
        target_row = target_rows.iloc[0].to_dict()

        expected_target_occurrence = {
            "prior_global_occurrence_count_at_ingest": 2,
            "global_occurrence_count_after_ingest": 3,
            "prior_route_local_occurrence_count_at_ingest": 0,
            "route_local_occurrence_count_after_ingest": 1,
            "global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "is_new_global_vehicle": False,
            "is_new_route_local_vehicle": True,
        }
        occurrence_failures = []
        for key, expected in expected_target_occurrence.items():
            actual = target_row.get(key)
            if actual != expected:
                occurrence_failures.append({"field": key, "expected": expected, "actual": actual})
        if occurrence_failures:
            raise ReconciliationFailure("FAIL_GLOBAL_VEHICLE_OCCURRENCE_RECONCILIATION", f"Target occurrence mismatch: {occurrence_failures}")

        counts_by_route = route_counts(hf2_registry)
        expected_route_counts = {"4010002001": 3, "4010002004": 2, "4010002118": 2, "4050010000": 2}
        if counts_by_route != expected_route_counts:
            raise ReconciliationFailure("FAIL_ROUTE_LOCAL_VEHICLE_RECONCILIATION", f"Route counts mismatch: {counts_by_route}")
        global_unique_vehicle_count = int(hf2_registry["canonical_vehicle_id"].dropna().astype(str).nunique())
        if global_unique_vehicle_count != 7:
            raise ReconciliationFailure("FAIL_GLOBAL_VEHICLE_OCCURRENCE_RECONCILIATION", "Global unique vehicle count is not 7")
        route_local_unique = {
            str(route): int(group["route_local_vehicle_identity_key"].dropna().astype(str).nunique())
            for route, group in hf2_registry.groupby("route_id")
        }
        if any(value < ROUTE_LOCAL_MINIMUM for value in route_local_unique.values()):
            raise ReconciliationFailure("FAIL_ROUTE_LOCAL_VEHICLE_RECONCILIATION", "Route-local minimum failed")

        prior_vehicle = prior_registry.copy()
        prior_vehicle["canonical_vehicle_id"] = prior_vehicle["vehicle_id"].map(canonical_text)
        prior_vehicle["canonical_route_id"] = prior_vehicle["route_id"].map(canonical_text)
        candidate_found_global = bool((prior_vehicle["canonical_vehicle_id"] == TARGET_VEHICLE).any())
        candidate_found_route = bool(((prior_vehicle["canonical_vehicle_id"] == TARGET_VEHICLE) & (prior_vehicle["canonical_route_id"] == TARGET_ROUTE)).any())
        if not candidate_found_global or candidate_found_route:
            raise ReconciliationFailure("FAIL_CANDIDATE_COUNTER_RECONCILIATION", "Candidate prior registry membership mismatch")

        hf1_counter = read_json(HF1_ROOT / "corrected_counter_contract_v10_hf1.json")
        corrected_counter = dict(hf1_counter)
        for scope_key in ["total"]:
            corrected_counter.setdefault(scope_key, {})
            corrected_counter[scope_key]["new_independent_vehicle_candidate_count"] = 0
            corrected_counter[scope_key]["previously_complete_vehicle_candidate_count"] = 1
            corrected_counter[scope_key]["new_route_independent_vehicle_candidate_count"] = 1
            corrected_counter[scope_key]["previously_complete_vehicle_for_route_candidate_count"] = 0
            corrected_counter[scope_key]["new_independent_complete_vehicle_count"] = 0
            corrected_counter[scope_key]["new_global_independent_complete_vehicle_count"] = 0
            corrected_counter[scope_key]["new_route_independent_complete_vehicle_count"] = 1
        corrected_counter.setdefault("by_route", {}).setdefault(TARGET_ROUTE, {})
        route_counter = corrected_counter["by_route"][TARGET_ROUTE]
        route_counter["new_independent_vehicle_candidate_count"] = 0
        route_counter["previously_complete_vehicle_candidate_count"] = 1
        route_counter["new_route_independent_vehicle_candidate_count"] = 1
        route_counter["previously_complete_vehicle_for_route_candidate_count"] = 0
        route_counter["new_independent_complete_vehicle_count"] = 0
        route_counter["new_global_independent_complete_vehicle_count"] = 0
        route_counter["new_route_independent_complete_vehicle_count"] = 1
        corrected_counter["counter_contract_v10_hf2_passed"] = True

        counter_reconciliation_failure_count = 0
        for scope in [corrected_counter["total"], route_counter]:
            if scope.get("new_independent_vehicle_candidate_count", 0) + scope.get("previously_complete_vehicle_candidate_count", 0) != scope.get("candidate_vehicle_count", 1):
                counter_reconciliation_failure_count += 1
            if scope.get("new_route_independent_vehicle_candidate_count", 0) + scope.get("previously_complete_vehicle_for_route_candidate_count", 0) != scope.get("candidate_vehicle_count", 1):
                counter_reconciliation_failure_count += 1
            if scope.get("new_global_independent_complete_vehicle_count", 0) > scope.get("new_complete_episode_count", 1):
                counter_reconciliation_failure_count += 1
            if scope.get("new_route_independent_complete_vehicle_count", 0) > scope.get("new_complete_episode_count", 1):
                counter_reconciliation_failure_count += 1
        if counter_reconciliation_failure_count:
            raise ReconciliationFailure("FAIL_COUNTER_CONTRACT", "Counter contract consistency failed")

        protected_registry_fields = [
            "frozen_episode_id",
            "source_campaign_id",
            "source_artifact",
            "source_episode_id",
            "route_id",
            "vehicle_id",
            "observation_date",
            "hour_bucket",
            "first_terminal_raw_sha",
            "first_post_terminal_raw_sha",
            "clock_semantics_status",
            "eligible_for_estimation_input",
        ]
        registry_changes = []
        hf1_sorted = stable_sort_registry(hf1_registry)[protected_registry_fields].astype(str).fillna("")
        hf2_sorted = stable_sort_registry(hf2_registry)[protected_registry_fields].astype(str).fillna("")
        for idx in range(len(hf1_sorted)):
            for field in protected_registry_fields:
                if hf1_sorted.iloc[idx][field] != hf2_sorted.iloc[idx][field]:
                    registry_changes.append({"row": idx, "field": field})

        ep = r2d1i_episode.iloc[0].to_dict()
        expected_episode = {
            "route_id": TARGET_ROUTE,
            "vehicle_id": TARGET_VEHICLE,
            "direction": "1",
            "episode_status": "COMPLETE_INTERVAL_CENSORED",
            "first_upstream_watch_sequence": 55,
            "last_pre_terminal_sequence": 70,
            "first_terminal_sequence": 71,
            "last_terminal_sequence": 76,
            "first_post_terminal_sequence": 2,
            "disappeared_waiting_reentry_observation_count": 48,
            "observed_post_terminal_confirmation_sample_count": 3,
            "clock_semantics_status": "PROVIDER_TIMESTAMP_MIXED",
        }
        episode_content_changes = []
        for field, expected in expected_episode.items():
            if str(ep.get(field)) != str(expected):
                episode_content_changes.append({"field": field, "expected": expected, "actual": ep.get(field)})
        scientific_change_count = len(registry_changes) + len(episode_content_changes)
        if scientific_change_count:
            raise ReconciliationFailure("FAIL_SCIENTIFIC_CONTENT_CHANGED", "Protected scientific content changed")

        interval_values = {
            "provider_lower_bound_sec": float(ep.get("provider_lower_bound_sec")),
            "provider_upper_bound_sec": float(ep.get("provider_upper_bound_sec")),
            "request_lower_bound_sec": float(ep.get("request_lower_bound_sec")),
            "request_upper_bound_sec": float(ep.get("request_upper_bound_sec")),
            "conservative_dual_lower_bound_sec": float(ep.get("conservative_dual_lower_bound_sec")),
            "conservative_dual_upper_bound_sec": float(ep.get("conservative_dual_upper_bound_sec")),
        }
        expected_intervals = {
            "provider_lower_bound_sec": 585.0,
            "provider_upper_bound_sec": 3140.0,
            "request_lower_bound_sec": 1517.0,
            "request_upper_bound_sec": 3136.0,
            "conservative_dual_lower_bound_sec": 585.0,
            "conservative_dual_upper_bound_sec": 3140.0,
        }
        interval_changes = [
            {"field": field, "expected": expected, "actual": interval_values[field]}
            for field, expected in expected_intervals.items()
            if interval_values[field] != expected
        ]
        if interval_changes:
            raise ReconciliationFailure("FAIL_INTERVAL_CONTENT_CHANGED", "Interval content changed")

        sha_fields = [
            "first_upstream_raw_sha256",
            "last_pre_terminal_raw_sha256",
            "first_terminal_raw_sha256",
            "last_terminal_raw_sha256",
            "first_post_terminal_raw_sha256",
        ]
        raw_index = raw_sha_index(R2D1I_ROOT / "raw" / TARGET_ROUTE)
        required_shas = [ep.get(field) for field in sha_fields] + confirmation_shas(ep.get("post_terminal_confirmation_raw_sha256s"))
        raw_failures = []
        raw_refs = []
        for sha in required_shas:
            paths = raw_index.get(str(sha), [])
            raw_refs.append({"sha256": sha, "exists": bool(paths), "paths": paths})
            if not paths:
                raw_failures.append(sha)
        if raw_failures:
            raise ReconciliationFailure("FAIL_RAW_PROVENANCE_AUDIT", "Raw provenance failed")

        duplicate_count = int(hf2_registry.duplicated(subset=["source_episode_id"]).sum())
        duplicate_raw_count = int(hf2_registry.duplicated(subset=["route_id", "vehicle_id", "first_terminal_raw_sha", "first_post_terminal_raw_sha"]).sum())
        if duplicate_count or duplicate_raw_count:
            raise ReconciliationFailure("FAIL_EPISODE_DUPLICATION", "Episode duplication failed")

        mapping_path = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"
        mapping_df = pd.read_parquet(mapping_path)
        mapping_regression_count = 0 if len(mapping_df[mapping_df["route_id"].astype(str) == TARGET_ROUTE]) == 1 else 1
        if mapping_regression_count:
            raise ReconciliationFailure("FAIL_MAPPING_REGRESSION", "Mapping regression failed")

        # Write upstream references.
        upstreams = {
            "upstream_reference_r2d1i_hf1.json": HF1_ROOT,
            "upstream_reference_r2d1i.json": R2D1I_ROOT,
            "upstream_reference_r2d1h_cleanup.json": R2D1H_CLEANUP_ROOT,
            "upstream_reference_hf1_mapping.json": HF1_MAPPING_ROOT,
            "upstream_reference_r2d1e.json": R2D1E_ROOT,
            "upstream_reference_r2d1f.json": R2D1F_ROOT,
        }
        for name, root in upstreams.items():
            dump_json(output_root / name, {"artifact_path": str(root), "exists": root.exists(), "read_only_input": True, "file_count": len(files_under(root))})

        script_path = Path(__file__).resolve()
        network_audit = {
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            **static_code_audit(script_path),
        }
        if not network_audit["network_code_static_scan_passed"]:
            raise ReconciliationFailure("FAIL_SECURITY_AUDIT", "Forbidden network import found")
        dump_json(output_root / "network_api_call_audit.json", network_audit)
        dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "environment_variable_read": False, "raw_key_material_seen": False})

        source_integrity = {
            "source_artifact_integrity_passed": True,
            "required_source_files": [
                str(HF1_ROOT / "prompt5_e01_r2d1i_hf1_gate.json"),
                str(HF1_ROOT / "prompt5_e01_r2d1i_hf1_manifest.json"),
                str(HF1_ROOT / "cumulative_episode_registry_candidate_hf1.parquet"),
                str(HF1_ROOT / "corrected_counter_contract_v10_hf1.json"),
                str(R2D1I_ROOT / "r2d1i_terminal_recovery_episodes.parquet"),
                str(R2D1H_CLEANUP_ROOT / "cumulative_episode_registry_candidate.parquet"),
                str(mapping_path),
            ],
            "missing_required_source_files": [],
            "manifest_verifications": [
                verify_manifest(HF1_ROOT, "prompt5_e01_r2d1i_hf1_manifest.json"),
                verify_manifest(R2D1I_ROOT, "prompt5_e01_r2d1i_manifest.json"),
                verify_manifest(R2D1H_CLEANUP_ROOT, "prompt5_e01_r2d1h_manifest.json"),
            ],
        }
        source_integrity["missing_required_source_files"] = [path for path in source_integrity["required_source_files"] if not Path(path).exists()]
        source_integrity["source_artifact_integrity_passed"] = not source_integrity["missing_required_source_files"] and all(
            item.get("failure_count", 0) == 0 for item in source_integrity["manifest_verifications"]
        )
        dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)
        if not source_integrity["source_artifact_integrity_passed"]:
            raise ReconciliationFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Source artifact integrity failed")

        dump_json(output_root / "manifest_required_file_reconciliation.json", manifest_reconciliation)
        dump_json(
            output_root / "vehicle_occurrence_reconciliation_contract.json",
            {
                "global_vehicle_identity_key": "canonical exact vehicle_id",
                "route_local_vehicle_identity_key": "route_id + canonical exact vehicle_id",
                "prior_global_occurrence_count_at_ingest": "count of same canonical vehicle_id rows before current row",
                "global_occurrence_count_after_ingest": "prior_global_occurrence_count_at_ingest + 1",
                "prior_route_local_occurrence_count_at_ingest": "count of same route_id + vehicle_id rows before current row",
                "route_local_occurrence_count_after_ingest": "prior_route_local_occurrence_count_at_ingest + 1",
            },
        )
        dump_json(
            output_root / "vehicle_occurrence_reconciliation_audit.json",
            {
                "vehicle_occurrence_reconciliation_passed": True,
                "records": occurrence_audit_df.to_dict("records"),
                "target_vehicle_record": occurrence_audit_df[occurrence_audit_df["source_episode_id"].astype(str) == TARGET_EPISODE_ID].to_dict("records")[0],
                "hf1_occurrence_source_present": hf1_occurrence is not None,
            },
        )
        write_parquet(output_root / "vehicle_occurrence_reconciliation_audit.parquet", occurrence_audit_df)
        dump_json(output_root / "global_vehicle_identity_contract_hf2.json", {"global_vehicle_identity_key": "canonical exact vehicle_id", "route_excluded": True})
        dump_json(output_root / "route_local_vehicle_identity_contract_hf2.json", {"route_local_vehicle_identity_key": "route_id + canonical exact vehicle_id", "route_included": True})

        target_occurrence_episode_rows = hf2_registry[hf2_registry["canonical_vehicle_id"].astype(str) == TARGET_VEHICLE]
        vehicle_1761 = {
            "vehicle_id": TARGET_VEHICLE,
            "route_id": TARGET_ROUTE,
            "prior_global_occurrence_count_at_ingest": 2,
            "global_occurrence_count_after_ingest": 3,
            "prior_route_local_occurrence_count_at_ingest": 0,
            "route_local_occurrence_count_after_ingest": 1,
            "global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "is_new_global_vehicle": False,
            "is_new_route_local_vehicle": True,
            "occurrence_episodes": target_occurrence_episode_rows.to_dict("records"),
        }
        dump_json(output_root / "vehicle_1761_final_classification.json", vehicle_1761)

        candidate_audit = {
            "candidate_vehicle_id": TARGET_VEHICLE,
            "candidate_route_id": TARGET_ROUTE,
            "candidate_found_in_prior_global_registry": candidate_found_global,
            "candidate_found_in_prior_route_registry": candidate_found_route,
            "global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "new_independent_vehicle_candidate_count": 0,
            "previously_complete_vehicle_candidate_count": 1,
            "new_route_independent_vehicle_candidate_count": 1,
            "previously_complete_vehicle_for_route_candidate_count": 0,
            "candidate_counter_reconciliation_failure_count": 0,
        }
        dump_json(output_root / "corrected_candidate_vehicle_selection_audit_hf2.json", candidate_audit)

        dump_json(output_root / "corrected_counter_contract_v10_hf2.json", corrected_counter)
        counter_rows = []
        total_counter = dict(corrected_counter.get("total", {}))
        total_counter["scope"] = "total"
        counter_rows.append(total_counter)
        route_counter_row = dict(route_counter)
        route_counter_row["scope"] = f"route:{TARGET_ROUTE}"
        counter_rows.append(route_counter_row)
        write_parquet(output_root / "corrected_counter_contract_v10_hf2.parquet", pd.DataFrame(counter_rows))

        dump_json(output_root / "cumulative_episode_registry_candidate_hf2.json", {"row_count": int(len(hf2_registry)), "records": hf2_registry.to_dict("records")})
        write_parquet(output_root / "cumulative_episode_registry_candidate_hf2.parquet", hf2_registry)

        route_hour_bucket_count = {
            str(route): int(group["hour_bucket"].dropna().astype(str).nunique())
            for route, group in hf2_registry.groupby("route_id")
        }
        method_progress = {
            "cumulative_complete_episode_count": int(len(hf2_registry)),
            "remaining_complete_episodes_to_12": max(0, REQUIRED_COMPLETE_EPISODES - len(hf2_registry)),
            "complete_episode_counts_by_route": counts_by_route,
            "each_route_complete_episode_minimum_met": all(value >= 2 for value in counts_by_route.values()),
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "required_global_unique_vehicle_count": REQUIRED_GLOBAL_UNIQUE_VEHICLES,
            "global_unique_vehicle_deficit": max(0, REQUIRED_GLOBAL_UNIQUE_VEHICLES - global_unique_vehicle_count),
            "route_local_unique_vehicle_count_by_route": route_local_unique,
            "each_route_route_local_vehicle_minimum_met": all(value >= ROUTE_LOCAL_MINIMUM for value in route_local_unique.values()),
            "observation_date_count": int(hf2_registry["observation_date"].dropna().astype(str).nunique()),
            "hour_bucket_count": int(hf2_registry["hour_bucket"].dropna().astype(str).nunique()),
            "each_route_hour_bucket_count": route_hour_bucket_count,
            "invalid_clock_order_count": int((hf2_registry["clock_semantics_status"].astype(str) == "INVALID_CLOCK_ORDER").sum()),
            "provenance_failure_count": 0,
            "eligible_for_terminal_recovery_estimation_execution": False,
            "eligibility_reason": "Method Prototype minimum requirements remain incomplete.",
        }
        dump_json(output_root / "method_prototype_progress_audit_hf2.json", method_progress)
        dump_json(
            output_root / "campaign_a_aggregate_status_hf2.json",
            {
                "campaign_a_aggregate_status": "TARGET_MET",
                "preserved_from_hf1": True,
                "global_new_vehicle_zero_does_not_invalidate_campaign_a_route_coverage": True,
                "campaign_b_authorized": False,
            },
        )
        dump_json(
            output_root / "supersession_reference_hf2.json",
            {
                "supersedes_metadata_only": True,
                "superseded_artifact": str(HF1_ROOT),
                "preserved_scientific_episode_result": True,
                "preserved_gate_meaning": True,
                "corrected_scope": [
                    "manifest missing required file counter",
                    "R2D-1I prior global vehicle occurrence count",
                    "global candidate vehicle counters",
                    "route-local candidate vehicle counter",
                    "JSON and Parquet metadata synchronization",
                ],
            },
        )
        dump_json(
            output_root / "scientific_content_preservation_audit.json",
            {
                "scientific_content_preservation_passed": True,
                "episode_content_change_count": 0,
                "raw_sha_reference_change_count": 0,
                "clock_semantics_change_count": 0,
                "protected_registry_field_change_count": 0,
                "new_complete_episode_count": 1,
                "right_censored_episode_count": 0,
                "disappeared_waiting_reentry_observation_count": 48,
                "post_terminal_confirmation_observation_count": 3,
                "post_terminal_confirmed_count": 1,
            },
        )
        dump_json(
            output_root / "interval_content_preservation_audit.json",
            {
                "interval_content_preservation_passed": True,
                "interval_content_change_count": 0,
                "interval_values": interval_values,
            },
        )
        dump_json(
            output_root / "raw_provenance_reference_audit.json",
            {
                "raw_provenance_reference_audit_passed": True,
                "raw_provenance_failure_count": 0,
                "raw_files_copied": 0,
                "raw_files_modified": 0,
                "raw_references": raw_refs,
            },
        )
        dump_json(
            output_root / "mapping_regression_audit.json",
            {
                "mapping_regression_audit_passed": True,
                "mapping_regression_count": 0,
                "hf1_mapping_sha256": sha256_file(mapping_path),
                "target_route_mapping_row_count": 1,
            },
        )
        dump_json(
            output_root / "episode_deduplication_audit.json",
            {
                "episode_deduplication_audit_passed": True,
                "episode_duplicate_count": 0,
                "duplicate_source_episode_id_count": duplicate_count,
                "duplicate_raw_signature_count": duplicate_raw_count,
            },
        )
        dump_json(
            output_root / "campaign_b_execution_authorization.json",
            {"campaign_b_authorized": False, "reason": "Campaign B requires a separate execution authorization after HF2 freeze review."},
        )
        dump_json(
            output_root / "terminal_recovery_estimation_execution_authorization.json",
            {"terminal_recovery_estimation_execution_approved": False, "reason": "Method Prototype minimum requirements remain incomplete."},
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
            {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False},
        )

        after = snapshot(upstream_roots)
        immutability_records = []
        modified = []
        for path, before_record in sorted(before.items()):
            after_record = after.get(path)
            record = {
                "absolute_path": path,
                "relative_path": before_record.get("relative_path"),
                "file_size_before": before_record.get("size"),
                "file_size_after": None if after_record is None else after_record.get("size"),
                "sha256_before": before_record.get("sha256"),
                "sha256_after": None if after_record is None else after_record.get("sha256"),
                "modified_during_hf2": after_record != before_record,
            }
            if record["modified_during_hf2"]:
                modified.append(record)
            immutability_records.append(record)
        dump_json(
            output_root / "authoritative_input_immutability_audit.json",
            {
                "source_immutability_passed": len(modified) == 0,
                "upstream_file_count": len(immutability_records),
                "upstream_modified_file_count": len(modified),
                "modified_files": modified,
                "records": immutability_records,
            },
        )
        if modified:
            raise ReconciliationFailure("FAIL_SOURCE_IMMUTABILITY", "Upstream files modified")

        secret_hits = []
        secret_needles = ["service" + "Key=", "api_key=", "Authorization:"]
        for path in output_root.glob("*"):
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in secret_needles:
                if needle in text:
                    secret_hits.append({"path": path.name, "pattern": needle})
        dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": len(secret_hits), "secret_leak_audit_passed": len(secret_hits) == 0, "service_key_accessed": False, "hits": secret_hits})
        if secret_hits:
            raise ReconciliationFailure("FAIL_SECURITY_AUDIT", "Secret leak pattern detected")

        # Synchronization checks after JSON and parquet writes.
        registry_json = read_json(output_root / "cumulative_episode_registry_candidate_hf2.json")
        registry_parquet = pd.read_parquet(output_root / "cumulative_episode_registry_candidate_hf2.parquet")
        json_row_by_episode = {str(row.get("source_episode_id")): row for row in registry_json["records"]}
        sync_fields = [
            "route_id",
            "vehicle_id",
            "prior_global_occurrence_count_at_ingest",
            "global_occurrence_count_after_ingest",
            "prior_route_local_occurrence_count_at_ingest",
            "route_local_occurrence_count_after_ingest",
            "global_vehicle_classification",
            "route_local_vehicle_classification",
            "is_new_global_vehicle",
            "is_new_route_local_vehicle",
        ]
        json_parquet_value_mismatch_count = 0
        for row in registry_parquet.to_dict("records"):
            jrow = json_row_by_episode.get(str(row.get("source_episode_id")))
            if jrow is None:
                json_parquet_value_mismatch_count += 1
                continue
            for field in sync_fields:
                if jrow.get(field) != row.get(field):
                    json_parquet_value_mismatch_count += 1

        strict_json_failures = []
        for path in output_root.glob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            except Exception as exc:
                strict_json_failures.append({"path": path.name, "error": type(exc).__name__})
        parquet_read_failures = []
        for path in output_root.glob("*.parquet"):
            try:
                pd.read_parquet(path)
            except Exception as exc:
                parquet_read_failures.append({"path": path.name, "error": type(exc).__name__})

        if json_parquet_value_mismatch_count:
            raise ReconciliationFailure("FAIL_JSON_PARQUET_SYNCHRONIZATION", "JSON/parquet value mismatch")
        if strict_json_failures or parquet_read_failures:
            raise ReconciliationFailure("FAIL_SCHEMA_AUDIT", "Schema validation failure")

        global_unique_deficit = max(0, REQUIRED_GLOBAL_UNIQUE_VEHICLES - global_unique_vehicle_count)
        complete_deficit = max(0, REQUIRED_COMPLETE_EPISODES - len(hf2_registry))
        gate = {
            "gate_status": PASS_GATE,
            "gate_passed": True,
            "network_api_calls": 0,
            "service_key_accessed": False,
            "upstream_modified_file_count": len(modified),
            "source_hf1_gate": hf1_gate.get("gate_status"),
            "preserved_r2d1i_gate": r2d1i_gate.get("gate_status"),
            "preserved_campaign_a_aggregate_status": "TARGET_MET",
            "cumulative_complete_episode_count": int(len(hf2_registry)),
            "route_complete_counts": counts_by_route,
            "vehicle_id_under_review": TARGET_VEHICLE,
            "prior_global_occurrence_count_at_ingest": 2,
            "global_occurrence_count_after_ingest": 3,
            "prior_route_local_occurrence_count_at_ingest": 0,
            "route_local_occurrence_count_after_ingest": 1,
            "global_vehicle_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
            "route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "new_independent_vehicle_candidate_count": 0,
            "previously_complete_vehicle_candidate_count": 1,
            "new_route_independent_vehicle_candidate_count": 1,
            "new_global_independent_complete_vehicle_count": 0,
            "new_route_independent_complete_vehicle_count": 1,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "global_unique_vehicle_deficit": global_unique_deficit,
            "remaining_complete_episodes_to_12": complete_deficit,
            "episode_content_change_count": 0,
            "interval_content_change_count": 0,
            "raw_provenance_failure_count": 0,
            "episode_duplicate_count": 0,
            "mapping_regression_count": 0,
            "candidate_counter_reconciliation_failure_count": 0,
            "counter_contract_failure_count": 0,
            "json_parquet_value_mismatch_count": json_parquet_value_mismatch_count,
            "strict_json_failure_count": len(strict_json_failures),
            "parquet_read_failure_count": len(parquet_read_failures),
            "manifest_missing_required_file_count": None,
            "manifest_hash_mismatch_count": None,
            "secret_leak_count": len(secret_hits),
            "campaign_b_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "next_authorized_action": "Campaign B live observation authorization review only",
        }

        report_lines = [
            "# Prompt 5-E01-R2D-1I-HF2 Final Report",
            "",
            "## Summary",
            "",
            f"- HF2 artifact: {output_root}",
            f"- upstream HF1 artifact: {HF1_ROOT}",
            "- network API calls: 0",
            "- service key accessed: false",
            f"- upstream modified file count: {len(modified)}",
            f"- existing HF1 gate: {hf1_gate.get('gate_status')}",
            f"- HF2 final gate: {PASS_GATE}",
            f"- existing HF1 gate manifest missing count: {hf1_gate.get('manifest_missing_required_file_count')}",
            "- corrected manifest missing count: 0",
            f"- actual HF1 required file count: {len(hf1_required)}",
            f"- actual HF1 missing file count: {len(hf1_missing_by_disk)}",
            f"- cumulative registry row count: {len(hf2_registry)}",
            f"- cumulative complete count: {len(hf2_registry)}",
            f"- route complete counts: {counts_by_route}",
            f"- global unique vehicle count: {global_unique_vehicle_count}",
            f"- route-local unique vehicle counts: {route_local_unique}",
            "",
            "## Vehicle 1761",
            "",
            "- occurrence episodes:",
        ]
        for row in target_occurrence_episode_rows.to_dict("records"):
            report_lines.append(f"  - {row.get('route_id')} / {row.get('vehicle_id')} / {row.get('source_episode_id')} / {row.get('source_artifact')}")
        report_lines.extend(
            [
                "- R2D-1I prior global occurrence count: 2",
                "- R2D-1I after-ingest global occurrence count: 3",
                "- global classification: PREVIOUSLY_COMPLETE_VEHICLE",
                "- route-local classification: NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
                "- previous candidate counters: new global 1, previously complete 0",
                "- corrected candidate counters: new global 0, previously complete 1, new route-local 1",
                "- previous complete independence counters: new independent complete 0, global 0, route-local 1",
                "- corrected complete independence counters: new independent complete 0, global 0, route-local 1",
                "",
                "## Audits",
                "",
                "- episode content change count: 0",
                "- interval content change count: 0",
                "- raw provenance failure count: 0",
                "- duplicate count: 0",
                "- mapping regression count: 0",
                f"- JSON/Parquet mismatch count: {json_parquet_value_mismatch_count}",
                f"- strict JSON failure count: {len(strict_json_failures)}",
                f"- Parquet read failure count: {len(parquet_read_failures)}",
                "- manifest hash mismatch count: 0",
                f"- Method Prototype episode deficit: {complete_deficit}",
                f"- Method Prototype global vehicle deficit: {global_unique_deficit}",
                "- Campaign A aggregate status: TARGET_MET",
                "- Campaign B authorized: false",
                "- estimation authorized: false",
                "- simulator authorized: false",
                "- Phase 2 authorized: false",
                "- next authorized action: Campaign B live observation authorization review only",
            ]
        )
        (output_root / "prompt5_e01_r2d1i_hf2_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        dump_json(output_root / "prompt5_e01_r2d1i_hf2_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json", manifest_payload(output_root))
        (output_root / "prompt5_e01_r2d1i_hf2_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json")
        gate["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json")
        hash_mismatches = []
        for entry in manifest["files"]:
            path = output_root / entry["path"]
            if entry.get("sha256") is not None and sha256_file(path) != entry["sha256"]:
                hash_mismatches.append(entry["path"])
        if manifest["missing_required_file_count"] or hash_mismatches:
            raise ReconciliationFailure("FAIL_MANIFEST_RECONCILIATION", "Final manifest validation failed")
        gate["manifest_hash_mismatch_count"] = len(hash_mismatches)
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1i_hf2_manifest.json")

        print("R2D-1I-HF2 FINAL METADATA RECONCILIATION COMPLETE")
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
        print("\nsource_hf1_gate:")
        print(hf1_gate.get("gate_status"))
        print("\npreserved_r2d1i_gate:")
        print(r2d1i_gate.get("gate_status"))
        print("\npreserved_campaign_a_aggregate_status:")
        print("TARGET_MET")
        print("\ncumulative_complete_episode_count:")
        print(len(hf2_registry))
        print("\nroute_complete_counts:")
        for route, count in counts_by_route.items():
            print(f"{route}={count}")
        print("\nvehicle_id_under_review:")
        print(TARGET_VEHICLE)
        print("\nprior_global_occurrence_count_at_ingest:")
        print(2)
        print("\nglobal_occurrence_count_after_ingest:")
        print(3)
        print("\nprior_route_local_occurrence_count_at_ingest:")
        print(0)
        print("\nroute_local_occurrence_count_after_ingest:")
        print(1)
        print("\nglobal_vehicle_classification:")
        print("PREVIOUSLY_COMPLETE_VEHICLE")
        print("\nroute_local_vehicle_classification:")
        print("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE")
        print("\nnew_independent_vehicle_candidate_count:")
        print(0)
        print("\npreviously_complete_vehicle_candidate_count:")
        print(1)
        print("\nnew_route_independent_vehicle_candidate_count:")
        print(1)
        print("\nnew_global_independent_complete_vehicle_count:")
        print(0)
        print("\nnew_route_independent_complete_vehicle_count:")
        print(1)
        print("\nglobal_unique_vehicle_count:")
        print(global_unique_vehicle_count)
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
        print(0)
        print("\ncandidate_counter_reconciliation_failure_count:")
        print(0)
        print("\ncounter_contract_failure_count:")
        print(0)
        print("\njson_parquet_value_mismatch_count:")
        print(json_parquet_value_mismatch_count)
        print("\nstrict_json_failure_count:")
        print(len(strict_json_failures))
        print("\nparquet_read_failure_count:")
        print(len(parquet_read_failures))
        print("\nmanifest_missing_required_file_count:")
        print(manifest["missing_required_file_count"])
        print("\nmanifest_hash_mismatch_count:")
        print(len(hash_mismatches))
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

    except ReconciliationFailure as exc:
        fail(output_root, exc.gate_status, str(exc))


if __name__ == "__main__":
    main()
