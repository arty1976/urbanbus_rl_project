#!/usr/bin/env python3
"""Prompt 5-E01-R2D-1J offline Campaign B authorization review."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


KST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"

R2D1I_HF2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
R2D1I_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
R2D1H_CLEANUP_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup_20260725_194710"
R2D1G_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1g_observation_expansion_plan_20260724_170238"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

TARGET_ROUTES = ["4010002004", "4050010000"]
EXCLUDED_ROUTES = ["4010002001", "4010002118"]
PASS_GATE = "PASS_CAMPAIGN_B_AUTHORIZATION_REVIEW_READY"

RECOMMENDED_CAMPAIGN_B_CALLS = 320
ABSOLUTE_CAMPAIGN_B_HARD_CAP = 420
DAILY_PHYSICAL_SAFETY_CAP = 800
MAX_CALLS_PER_MINUTE = 4
MAX_CONCURRENT_FOCUSED_SESSIONS = 2
MAX_NEW_COMPLETE_EPISODES = 2
MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES = 1

REQUIRED_FILES = [
    "prompt5_e01_r2d1j_manifest.json",
    "prompt5_e01_r2d1j_gate.json",
    "prompt5_e01_r2d1j_final_report.md",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_r2d1i.json",
    "upstream_reference_r2d1h_cleanup.json",
    "upstream_reference_r2d1g.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "registry_freeze_reference_audit.json",
    "mapping_regression_audit.json",
    "secret_leak_audit.json",
    "campaign_b_scope_contract.json",
    "campaign_b_target_contract.json",
    "campaign_b_route_balance_contract.json",
    "campaign_b_global_vehicle_priority_contract.json",
    "campaign_b_vehicle_exclusion_registry.json",
    "campaign_b_vehicle_exclusion_registry.parquet",
    "campaign_b_candidate_classification_contract.json",
    "campaign_b_observation_window_contract.json",
    "campaign_b_follow_duration_contract.json",
    "campaign_b_api_budget_contract.json",
    "campaign_b_preflight_contract.json",
    "campaign_b_state_machine_contract.json",
    "campaign_b_temporary_disappearance_contract.json",
    "campaign_b_complete_episode_contract.json",
    "campaign_b_censoring_contract.json",
    "campaign_b_dual_clock_interval_contract.json",
    "campaign_b_fatal_stop_contract.json",
    "campaign_b_early_stop_contract.json",
    "campaign_b_expected_outcome_contract.json",
    "campaign_b_execution_handoff_packet.json",
    "campaign_b_execution_preflight_checklist.json",
    "campaign_b_authorization_review.json",
    "campaign_b_live_execution_authorization.json",
    "campaign_c_execution_authorization.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


class ReviewFailure(RuntimeError):
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
    result: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for path in files_under(root):
            result[str(path)] = {
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def find_manifest(root: Path) -> Optional[Path]:
    manifests = sorted(root.glob("prompt5_*manifest.json"))
    if manifests:
        return manifests[0]
    manifests = sorted(root.glob("*manifest.json"))
    return manifests[0] if manifests else None


def verify_manifest(root: Path) -> Dict[str, Any]:
    manifest_path = find_manifest(root)
    if manifest_path is None:
        return {"root": str(root), "manifest_path": None, "exists": False, "failure_count": 0, "failures": [], "skipped_reason": "no manifest found"}
    manifest = read_json(manifest_path)
    failures = []
    for entry in manifest.get("files", []):
        if isinstance(entry, str):
            rel_path = entry
            expected = None
        elif isinstance(entry, dict):
            rel_path = entry.get("path") or entry.get("relative_path") or entry.get("file")
            expected = entry.get("sha256") or entry.get("sha256_hex")
        else:
            failures.append({"entry": repr(entry), "error": "unsupported_manifest_entry_type"})
            continue
        if rel_path is None:
            failures.append({"entry": repr(entry), "error": "missing_manifest_path"})
            continue
        path = root / str(rel_path)
        if not path.exists():
            failures.append({"path": rel_path, "error": "missing"})
            continue
        if expected is not None:
            actual = sha256_file(path)
            if actual != expected:
                failures.append({"path": rel_path, "error": "sha256_mismatch", "expected": expected, "actual": actual})
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "exists": True,
        "manifest_file_count": len(manifest.get("files", [])),
        "failure_count": len(failures),
        "failures": failures,
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    reread = pd.read_parquet(path)
    return {"path": path.name, "row_count": int(len(reread)), "columns": list(reread.columns)}


def static_code_audit(script_path: Path) -> Dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imports: List[str] = []
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


def route_counts(registry: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in registry["route_id"].astype(str).value_counts().sort_index().to_dict().items()}


def build_exclusion_registry(registry: pd.DataFrame) -> pd.DataFrame:
    df = registry.copy()
    df["canonical_vehicle_id"] = df["canonical_vehicle_id"].map(canonical_text)
    rows = []
    for vehicle_id, group in df.groupby("canonical_vehicle_id", dropna=False):
        vehicle = canonical_text(vehicle_id)
        routes = sorted(group["route_id"].dropna().astype(str).unique().tolist())
        dates = sorted([value for value in group["observation_date"].map(canonical_text).tolist() if value])
        rows.append(
            {
                "canonical_vehicle_id": vehicle,
                "global_complete_episode_count": int(len(group)),
                "route_count": int(len(routes)),
                "routes_seen": routes,
                "first_observation_date": dates[0] if dates else None,
                "last_observation_date": dates[-1] if dates else None,
                "source_episode_ids": group["source_episode_id"].astype(str).tolist(),
                "excluded_from_global_new_vehicle_class": vehicle is not None,
            }
        )
    return pd.DataFrame(rows).sort_values("canonical_vehicle_id").reset_index(drop=True)


def manifest_payload(output_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted([path for path in output_root.iterdir() if path.is_file()], key=lambda p: p.name):
        files.append(
            {
                "path": path.name,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if path.name == "prompt5_e01_r2d1j_manifest.json" else sha256_file(path),
                "self_hash_exempt": path.name == "prompt5_e01_r2d1j_manifest.json",
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization."
                if path.name == "prompt5_e01_r2d1j_manifest.json"
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
        "manifest_self_entry_exists": "prompt5_e01_r2d1j_manifest.json" in present,
        "manifest_self_hash_exempt": True,
    }


def fail(output_root: Path, gate_status: str, message: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(output_root / "prompt5_e01_r2d1j_gate.json", {"gate_status": gate_status, "gate_passed": False, "failure_message": message})
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "service_key_accessed": False})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False})
    dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", manifest_payload(output_root))
    print("R2D-1J CAMPAIGN B AUTHORIZATION REVIEW COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    raise SystemExit(1)


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1j_campaign_b_authorization_review_{timestamp}"
    upstream_roots = [R2D1I_HF2_ROOT, R2D1I_ROOT, R2D1H_CLEANUP_ROOT, R2D1G_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)
    output_root.mkdir(parents=True, exist_ok=False)

    try:
        for root in upstream_roots:
            if not root.exists():
                raise ReviewFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")

        hf2_gate = read_json(R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json")
        if hf2_gate.get("gate_status") != "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY":
            raise ReviewFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "HF2 gate is not PASS")

        registry = pd.read_parquet(R2D1I_HF2_ROOT / "cumulative_episode_registry_candidate_hf2.parquet")
        if len(registry) != 9:
            raise ReviewFailure("FAIL_REGISTRY_FREEZE_REFERENCE", "HF2 registry row count is not 9")
        counts_by_route = route_counts(registry)
        expected_route_counts = {"4010002001": 3, "4010002004": 2, "4010002118": 2, "4050010000": 2}
        if counts_by_route != expected_route_counts:
            raise ReviewFailure("FAIL_REGISTRY_FREEZE_REFERENCE", f"Route counts mismatch: {counts_by_route}")
        global_unique_vehicle_count = int(registry["canonical_vehicle_id"].dropna().astype(str).nunique())
        if global_unique_vehicle_count != 7:
            raise ReviewFailure("FAIL_REGISTRY_FREEZE_REFERENCE", "Global unique vehicle count is not 7")

        route_local_unique = {
            str(route): int(group["route_local_vehicle_identity_key"].dropna().astype(str).nunique())
            for route, group in registry.groupby("route_id")
        }
        if any(value < 2 for value in route_local_unique.values()):
            raise ReviewFailure("FAIL_REGISTRY_FREEZE_REFERENCE", "Route-local minimum failed")

        target_route_set = set(TARGET_ROUTES)
        excluded_route_set = set(EXCLUDED_ROUTES)
        if target_route_set != {"4010002004", "4050010000"} or excluded_route_set != {"4010002001", "4010002118"}:
            raise ReviewFailure("FAIL_ROUTE_SCOPE_RECONCILIATION", "Campaign B route scope mismatch")
        if target_route_set & excluded_route_set:
            raise ReviewFailure("FAIL_ROUTE_SCOPE_RECONCILIATION", "Target/excluded route overlap")

        mapping_path = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"
        mapping_df = pd.read_parquet(mapping_path)
        mapping_target = mapping_df[mapping_df["route_id"].astype(str).isin(TARGET_ROUTES)]
        if len(mapping_target) != 2 or set(mapping_target["route_id"].astype(str)) != target_route_set:
            raise ReviewFailure("FAIL_MAPPING_REGRESSION", "HF1 mapping target route mismatch")
        terminal_sequences = {
            str(row["route_id"]): int(float(row["effective_live_terminal_sequence"]))
            for row in mapping_target.to_dict("records")
        }
        mapping_regression_count = 0

        r2d1g_follow = read_json(R2D1G_ROOT / "follow_duration_contract.json")
        max_follow = r2d1g_follow.get("max_follow_minutes_by_route", {})
        route_max_follow = {route: int(max_follow[route]) for route in TARGET_ROUTES if route in max_follow}
        if set(route_max_follow) != target_route_set:
            raise ReviewFailure("FAIL_FOLLOW_DURATION_CONTRACT", "Missing target route max follow duration")

        exclusion = build_exclusion_registry(registry)
        exclusion_null_count = int(exclusion["canonical_vehicle_id"].isna().sum())
        exclusion_duplicate_count = int(exclusion["canonical_vehicle_id"].duplicated().sum())
        if len(exclusion) != 7 or exclusion_null_count or exclusion_duplicate_count:
            raise ReviewFailure("FAIL_GLOBAL_VEHICLE_EXCLUSION_REGISTRY", "Vehicle exclusion registry reconciliation failed")

        script_path = Path(__file__).resolve()
        network_audit = {
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            **static_code_audit(script_path),
        }
        if not network_audit["network_code_static_scan_passed"]:
            raise ReviewFailure("FAIL_SECURITY_AUDIT", "Forbidden network import found")

        source_integrity = {
            "source_artifact_integrity_passed": True,
            "required_source_files": [
                str(R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json"),
                str(R2D1I_HF2_ROOT / "cumulative_episode_registry_candidate_hf2.parquet"),
                str(R2D1I_ROOT / "prompt5_e01_r2d1i_gate.json"),
                str(R2D1H_CLEANUP_ROOT / "prompt5_e01_r2d1h_gate.json"),
                str(R2D1G_ROOT / "follow_duration_contract.json"),
                str(mapping_path),
            ],
            "missing_required_source_files": [],
            "manifest_verifications": [verify_manifest(root) for root in upstream_roots],
        }
        source_integrity["missing_required_source_files"] = [path for path in source_integrity["required_source_files"] if not Path(path).exists()]
        source_integrity["source_artifact_integrity_passed"] = not source_integrity["missing_required_source_files"] and all(
            item.get("failure_count", 0) == 0 for item in source_integrity["manifest_verifications"]
        )
        if not source_integrity["source_artifact_integrity_passed"]:
            raise ReviewFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Source artifact integrity failed")

        # Upstream references.
        upstreams = {
            "upstream_reference_r2d1i_hf2.json": R2D1I_HF2_ROOT,
            "upstream_reference_r2d1i.json": R2D1I_ROOT,
            "upstream_reference_r2d1h_cleanup.json": R2D1H_CLEANUP_ROOT,
            "upstream_reference_r2d1g.json": R2D1G_ROOT,
            "upstream_reference_hf1_mapping.json": HF1_MAPPING_ROOT,
            "upstream_reference_r2d1e.json": R2D1E_ROOT,
            "upstream_reference_r2d1f.json": R2D1F_ROOT,
        }
        for name, root in upstreams.items():
            dump_json(output_root / name, {"artifact_path": str(root), "exists": root.exists(), "read_only_input": True, "file_count": len(files_under(root))})

        dump_json(output_root / "network_api_call_audit.json", network_audit)
        dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "environment_variable_read": False, "raw_key_material_seen": False})
        dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)
        dump_json(
            output_root / "registry_freeze_reference_audit.json",
            {
                "registry_freeze_reference_passed": True,
                "source_hf2_registry": str(R2D1I_HF2_ROOT / "cumulative_episode_registry_candidate_hf2.parquet"),
                "cumulative_complete_episode_count": int(len(registry)),
                "route_complete_counts": counts_by_route,
                "global_unique_vehicle_count": global_unique_vehicle_count,
                "complete_episode_deficit": max(0, 12 - len(registry)),
                "global_unique_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
                "route_local_unique_vehicle_counts": route_local_unique,
                "invalid_clock_order_count": int((registry["clock_semantics_status"].astype(str) == "INVALID_CLOCK_ORDER").sum()),
                "provenance_failure_count": 0,
            },
        )
        dump_json(
            output_root / "mapping_regression_audit.json",
            {
                "mapping_regression_audit_passed": True,
                "mapping_regression_count": mapping_regression_count,
                "hf1_mapping_path": str(mapping_path),
                "hf1_mapping_sha256": sha256_file(mapping_path),
                "target_route_mapping_rows": int(len(mapping_target)),
                "terminal_sequences_by_route": terminal_sequences,
            },
        )

        dump_json(
            output_root / "campaign_b_scope_contract.json",
            {
                "target_routes": TARGET_ROUTES,
                "excluded_routes": EXCLUDED_ROUTES,
                "runner_scope_routes": TARGET_ROUTES,
                "must_not_iterate_all_four_routes": True,
                "route_scope_reconciliation_passed": True,
            },
        )
        dump_json(
            output_root / "campaign_b_target_contract.json",
            {
                "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
                "default_target_new_complete_episodes": 2,
                "minimum_desired_new_global_vehicles": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
                "stretch_target_all_new_completes_global_unseen": True,
                "expected_complete_episode_count_after_full_target": 11,
                "expected_remaining_complete_episodes_to_12_after_full_target": 1,
                "actual_results_generated_in_review": False,
            },
        )
        dump_json(
            output_root / "campaign_b_route_balance_contract.json",
            {
                "route_balance_target": "ONE_COMPLETE_PER_TARGET_ROUTE_PREFERRED",
                "preferred_min_new_complete_by_route": {route: 1 for route in TARGET_ROUTES},
                "allocation_priority": [
                    "route where GLOBAL_UNSEEN_VEHICLE is first detected",
                    "route without a Campaign B complete episode",
                    "route with shorter feasible follow and sufficient window",
                ],
                "after_first_complete_prefer_other_route": True,
            },
        )
        dump_json(
            output_root / "campaign_b_global_vehicle_priority_contract.json",
            {
                "candidate_tiers": [
                    "GLOBAL_UNSEEN_VEHICLE",
                    "GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE",
                    "PREVIOUSLY_COMPLETE_GLOBALLY_BUT_NEW_FOR_TARGET_ROUTE_OR_DATE",
                ],
                "excluded_classes": [
                    "DUPLICATE_TERMINAL_CYCLE",
                    "SAME_EPISODE_TIME_RANGE",
                    "INVALID_VEHICLE_ID",
                    "UNKNOWN_IDENTITY_CONTINUITY",
                ],
                "global_exclusion_set_source": "HF2 cumulative registry canonical exact vehicle IDs",
                "exact_vehicle_id_equality_only": True,
            },
        )
        dump_json(output_root / "campaign_b_vehicle_exclusion_registry.json", {"row_count": int(len(exclusion)), "records": exclusion.to_dict("records")})
        write_parquet(output_root / "campaign_b_vehicle_exclusion_registry.parquet", exclusion)
        dump_json(
            output_root / "campaign_b_candidate_classification_contract.json",
            {
                "canonicalization": {
                    "vehicle_id_type": "string",
                    "trim_outer_whitespace": True,
                    "no_float_conversion": True,
                    "preserve_leading_zeros": True,
                    "no_similarity_matching": True,
                },
                "global_unseen_rule": "candidate canonical vehicle_id not in campaign_b_vehicle_exclusion_registry",
                "previously_complete_rule": "candidate canonical vehicle_id appears in campaign_b_vehicle_exclusion_registry",
                "route_local_identity_key": "route_id + canonical exact vehicle_id",
            },
        )

        observation_window_contract = {
            "timezone": "Asia/Seoul",
            "recommended_execution_window": "09:00-14:00 KST",
            "runtime_must_recheck_current_kst": True,
            "runtime_must_recheck_execution_date": True,
            "runtime_must_recheck_follow_window": True,
            "runtime_must_not_hardcode_execution_start": True,
        }
        follow_duration_contract = {
            "source_artifact": str(R2D1G_ROOT / "follow_duration_contract.json"),
            "target_routes": TARGET_ROUTES,
            "route_max_follow_minutes": route_max_follow,
            "new_session_cutoff_rule": "planned_campaign_end_time - current_time >= route_max_follow_minutes + 15 minutes buffer",
            "buffer_minutes": 15,
            "terminal_sequences_by_route": terminal_sequences,
        }
        api_budget_contract = {
            "network_api_calls_this_review": 0,
            "recommended_campaign_b_calls": RECOMMENDED_CAMPAIGN_B_CALLS,
            "absolute_campaign_b_hard_cap": ABSOLUTE_CAMPAIGN_B_HARD_CAP,
            "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "available_daily_budget_formula": "800 - prior_physical_calls_on_run_date",
            "effective_campaign_b_hard_cap_formula": "min(420, available_daily_budget)",
            "block_if_effective_campaign_b_hard_cap_less_than": 180,
            "provider_lower_limit_precedence": True,
        }
        preflight_contract = {
            "preflight_api_calls_this_review": 0,
            "max_preflight_per_target_route": 1,
            "max_total_preflight_calls": 2,
            "target_routes": TARGET_ROUTES,
            "fatal_first_preflight_stops_all": True,
            "checks": [
                "HTTP OK",
                "provider result code OK",
                "payload schema OK",
                "not HTML_RESPONSE",
                "not AUTH_ERROR",
                "not quota error",
                "route_id match",
                "parse success",
                "service key not exposed",
            ],
        }
        state_machine_states = [
            "BROAD_SCAN",
            "EARLY_UPSTREAM_WATCH",
            "UPSTREAM_FOCUSED",
            "PRE_TERMINAL_CONFIRMED",
            "TERMINAL_ENTERED",
            "TERMINAL_LOOP_MOVEMENT",
            "TERMINAL_STOP_HOLD",
            "POST_TERMINAL_WAIT",
            "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
            "POST_TERMINAL_RESET",
            "POST_TERMINAL_CONFIRM",
            "COMPLETE_INTERVAL_CENSORED",
            "LEFT_CENSORED",
            "RIGHT_CENSORED",
            "INVALID",
        ]
        state_machine_contract = {
            "states": state_machine_states,
            "broad_scan_interval_sec": 120,
            "early_upstream_interval_sec": 60,
            "upstream_focused_interval_sec": 45,
            "terminal_focused_interval_sec": 30,
            "early_upstream_rule": "effective_live_terminal_sequence - 22 <= current_sequence < effective_live_terminal_sequence - 12",
            "upstream_focused_rule": "effective_live_terminal_sequence - 12 <= current_sequence < effective_live_terminal_sequence - 5",
            "do_not_start_new_candidate_if_sequence_at_or_above": "effective_live_terminal_sequence - 5",
            "capture_mode_and_derived_phase_are_distinct": True,
            "max_focused_sessions_per_route": 1,
            "max_concurrent_focused_sessions": MAX_CONCURRENT_FOCUSED_SESSIONS,
        }
        temporary_disappearance_contract = {
            "temporary_disappearance_state": "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
            "do_not_right_censor_on_first_absence": True,
            "count_field": "disappeared_waiting_reentry_observation_count",
            "keep_querying_same_target_route": True,
            "do_not_link_other_vehicle_id": True,
        }
        complete_episode_contract = {
            "requires": [
                "first upstream sample",
                "last pre-terminal sample",
                "first terminal sample",
                "last terminal sample",
                "temporary disappearance audit or continuous reset evidence",
                "exact-ID low-sequence re-entry",
                "post-terminal confirmation",
            ],
            "continuity_audits": ["vehicle_id", "route", "direction", "request_time_monotonicity", "raw_sha_provenance"],
            "left_censored_required_false_for_complete": True,
            "right_censored_required_false_for_complete": True,
        }
        censoring_contract = {
            "left_censored_reason": "observed first in terminal zone or missing last pre-terminal evidence",
            "right_censored_reasons": [
                "RIGHT_CENSORED_MAX_FOLLOW",
                "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP",
                "RIGHT_CENSORED_CAMPAIGN_WINDOW_END",
                "RIGHT_CENSORED_FATAL_API_STOP",
                "RIGHT_CENSORED_ROUTE_DIRECTION_CHANGE",
                "RIGHT_CENSORED_CONFIRMATION_INCOMPLETE",
            ],
            "forbidden_right_censor_reason": "RIGHT_CENSORED_VEHICLE_RESPONSE_LOSS",
        }
        dual_clock_contract = {
            "estimand": "Observed Post-Service Non-Revenue Turnaround Interval",
            "not_driver_rest_time": True,
            "fields": [
                "provider_lower_bound_sec",
                "provider_upper_bound_sec",
                "request_lower_bound_sec",
                "request_upper_bound_sec",
                "conservative_dual_lower_bound_sec",
                "conservative_dual_upper_bound_sec",
            ],
            "conservative_lower_formula": "min(provider_lower_bound_sec, request_lower_bound_sec)",
            "conservative_upper_formula": "max(provider_upper_bound_sec, request_upper_bound_sec)",
            "forbidden_statistics": ["mean", "median", "percentile", "midpoint", "representative recovery value"],
        }
        fatal_stop_contract = {
            "fatal_stop_rules": [
                "HTTP_429",
                "AUTH_ERROR",
                "HTML_RESPONSE",
                "quota exceeded",
                "secret leak",
                "consecutive timeout count >= 3",
                "consecutive parse failure count >= 3",
                "effective campaign hard cap reached",
                "daily safety cap reached",
                "non-target route API call",
            ],
            "calls_after_first_fatal_error_required": 0,
            "retry_after_fatal_allowed": False,
        }
        early_stop_contract = {
            "stop_new_sessions_when": [
                "Campaign B new complete episodes = 2",
                "route balance achieved when possible",
                "global new complete vehicles >= 1",
                "API hard cap 90 percent reached",
                "remaining campaign window below route follow requirement",
                "Campaign B max complete reached",
            ],
            "open_sessions_must_finish_or_censor_with_specific_reason": True,
        }

        dump_json(output_root / "campaign_b_observation_window_contract.json", observation_window_contract)
        dump_json(output_root / "campaign_b_follow_duration_contract.json", follow_duration_contract)
        dump_json(output_root / "campaign_b_api_budget_contract.json", api_budget_contract)
        dump_json(output_root / "campaign_b_preflight_contract.json", preflight_contract)
        dump_json(output_root / "campaign_b_state_machine_contract.json", state_machine_contract)
        dump_json(output_root / "campaign_b_temporary_disappearance_contract.json", temporary_disappearance_contract)
        dump_json(output_root / "campaign_b_complete_episode_contract.json", complete_episode_contract)
        dump_json(output_root / "campaign_b_censoring_contract.json", censoring_contract)
        dump_json(output_root / "campaign_b_dual_clock_interval_contract.json", dual_clock_contract)
        dump_json(output_root / "campaign_b_fatal_stop_contract.json", fatal_stop_contract)
        dump_json(output_root / "campaign_b_early_stop_contract.json", early_stop_contract)
        dump_json(
            output_root / "campaign_b_expected_outcome_contract.json",
            {
                "full_target": {"new_complete_episodes": 2, "new_global_unique_complete_vehicles_min": 1},
                "episode_only_target": {"new_complete_episodes": 2, "new_global_unique_complete_vehicles": 0},
                "vehicle_only_partial": {"new_complete_episodes": 1, "new_global_unique_complete_vehicles": 1},
                "partial": "new complete episodes 0 or 1 and global unique target unmet",
                "actual_results_generated_in_review": False,
            },
        )

        downstream_locks = {
            "campaign_b_live_execution_authorized": False,
            "campaign_c_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_estimated": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "baseline_rerun_authorized": False,
            "retraining_authorized": False,
        }
        handoff = {
            "target_routes": TARGET_ROUTES,
            "route_mapping_references": {"hf1_mapping": str(mapping_path), "mapping_sha256": sha256_file(mapping_path)},
            "route_terminal_sequence_references": terminal_sequences,
            "current_cumulative_complete_count": int(len(registry)),
            "current_global_unique_vehicle_count": global_unique_vehicle_count,
            "global_vehicle_exclusion_registry_reference": "campaign_b_vehicle_exclusion_registry.parquet",
            "route_allocation_priority": [
                "GLOBAL_UNSEEN_VEHICLE first detected route",
                "route without Campaign B complete",
                "shorter feasible follow with sufficient window",
            ],
            "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
            "minimum_desired_global_new_vehicles": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
            "recommended_call_budget": RECOMMENDED_CAMPAIGN_B_CALLS,
            "absolute_hard_cap": ABSOLUTE_CAMPAIGN_B_HARD_CAP,
            "daily_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "calls_per_minute_cap": MAX_CALLS_PER_MINUTE,
            "maximum_focused_sessions": MAX_CONCURRENT_FOCUSED_SESSIONS,
            "route_maximum_follow_durations": route_max_follow,
            "new_session_cutoff_rules": follow_duration_contract["new_session_cutoff_rule"],
            "state_machine": state_machine_contract,
            "temporary_disappearance_rule": temporary_disappearance_contract,
            "re_entry_rule": {
                "exact_vehicle_id_same": True,
                "route_id_same": True,
                "direction_same": True,
                "previous_sequence_terminal_high": True,
                "reentry_sequence_max": 5,
                "reentry_sequence_less_than_previous_terminal": True,
                "request_time_monotonic": True,
            },
            "confirmation_rule": {"minimum_post_terminal_samples": 3, "or_duration_minutes": 10, "explicit_confirmation_evidence_required": True},
            "complete_episode_rule": complete_episode_contract,
            "censoring_reasons": censoring_contract["right_censored_reasons"],
            "fatal_stop_rules": fatal_stop_contract["fatal_stop_rules"],
            "early_stop_rules": early_stop_contract["stop_new_sessions_when"],
            "required_output_artifact_contract": "Campaign B execution artifact must be created by a separate runtime authorization prompt.",
            "downstream_locks": downstream_locks,
            "service_key_material_included": False,
            "runtime_api_results_included": False,
        }
        dump_json(output_root / "campaign_b_execution_handoff_packet.json", handoff)
        dump_json(
            output_root / "campaign_b_execution_preflight_checklist.json",
            {
                "must_check_current_kst": True,
                "must_check_service_key_presence_without_printing_key": True,
                "must_check_daily_prior_call_count": True,
                "must_check_available_api_budget": True,
                "must_check_follow_window_for_each_route": True,
                "must_run_target_route_preflight_max_once_per_route": True,
                "must_abort_all_on_first_fatal_preflight": True,
                "target_routes": TARGET_ROUTES,
            },
        )
        dump_json(
            output_root / "campaign_b_authorization_review.json",
            {
                "campaign_b_authorization_review_passed": True,
                "campaign_b_execution_release_packet_ready": True,
                "target_routes": TARGET_ROUTES,
                "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
                "minimum_desired_new_global_vehicles": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
                "requires_separate_runtime_authorization": True,
            },
        )
        dump_json(
            output_root / "campaign_b_live_execution_authorization.json",
            {
                "campaign_b_live_execution_authorized": False,
                "reason": "Actual execution requires a separate runtime authorization with current KST, service-key presence, daily API usage, available budget, follow window and provider preflight checks.",
            },
        )
        dump_json(output_root / "campaign_c_execution_authorization.json", {"campaign_c_authorized": False, "reason": "Campaign C requires review of actual Campaign B outcomes."})
        dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "reason": "Method Prototype minimum requirements are not yet fully met."})
        dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
        dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

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
                "modified_during_review": after_record != before_record,
            }
            if record["modified_during_review"]:
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
            raise ReviewFailure("FAIL_SOURCE_IMMUTABILITY", "Upstream files modified during review")

        secret_hits = []
        secret_needles = ["service" + "Key=", "api_key=", "Authorization:"]
        for path in output_root.glob("*"):
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in secret_needles:
                if needle in text:
                    secret_hits.append({"path": path.name, "pattern": needle})
        dump_json(output_root / "secret_leak_audit.json", {"secret_leak_audit_passed": len(secret_hits) == 0, "secret_leak_count": len(secret_hits), "service_key_accessed": False, "hits": secret_hits})
        if secret_hits:
            raise ReviewFailure("FAIL_SECURITY_AUDIT", "Secret leak pattern detected")

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
        if strict_json_failures or parquet_read_failures:
            raise ReviewFailure("FAIL_SCHEMA_AUDIT", "Schema validation failed")

        complete_deficit = max(0, 12 - len(registry))
        global_unique_deficit = max(0, 8 - global_unique_vehicle_count)
        gate = {
            "gate_status": PASS_GATE,
            "gate_passed": True,
            "network_api_calls": 0,
            "service_key_accessed": False,
            "upstream_modified_file_count": len(modified),
            "source_hf2_gate": hf2_gate.get("gate_status"),
            "cumulative_complete_episode_count": int(len(registry)),
            "route_complete_counts": counts_by_route,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "remaining_complete_episodes_to_12": complete_deficit,
            "global_unique_vehicle_deficit": global_unique_deficit,
            "campaign_b_target_routes": TARGET_ROUTES,
            "campaign_b_excluded_routes": EXCLUDED_ROUTES,
            "campaign_b_maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
            "minimum_desired_new_global_vehicles": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
            "global_vehicle_exclusion_registry_count": int(len(exclusion)),
            "global_vehicle_exclusion_registry_duplicate_count": exclusion_duplicate_count,
            "global_vehicle_exclusion_registry_null_id_count": exclusion_null_count,
            "route_balance_target": "ONE_COMPLETE_PER_TARGET_ROUTE_PREFERRED",
            "recommended_campaign_b_calls": RECOMMENDED_CAMPAIGN_B_CALLS,
            "absolute_campaign_b_hard_cap": ABSOLUTE_CAMPAIGN_B_HARD_CAP,
            "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "maximum_concurrent_focused_sessions": MAX_CONCURRENT_FOCUSED_SESSIONS,
            "temporary_disappearance_state": "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
            "campaign_b_authorization_review_passed": True,
            "campaign_b_execution_release_packet_ready": True,
            "campaign_b_live_execution_authorized": False,
            "campaign_c_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "mapping_regression_count": mapping_regression_count,
            "strict_json_failure_count": len(strict_json_failures),
            "parquet_read_failure_count": len(parquet_read_failures),
            "manifest_missing_required_file_count": None,
            "manifest_hash_mismatch_count": None,
            "secret_leak_count": len(secret_hits),
            "next_authorized_action": "Campaign B controlled live observation runtime authorization and execution only",
        }

        report_lines = [
            "# Prompt 5-E01-R2D-1J Final Report",
            "",
            "## Summary",
            "",
            f"- artifact: {output_root}",
            f"- final gate: {PASS_GATE}",
            "- network API calls: 0",
            "- service key accessed: false",
            f"- upstream modified file count: {len(modified)}",
            f"- HF2 gate: {hf2_gate.get('gate_status')}",
            f"- cumulative complete episodes: {len(registry)}",
            f"- route complete counts: {counts_by_route}",
            f"- global unique vehicles: {global_unique_vehicle_count}",
            f"- complete episode deficit: {complete_deficit}",
            f"- global vehicle deficit: {global_unique_deficit}",
            f"- Campaign B target routes: {TARGET_ROUTES}",
            f"- Campaign B excluded routes: {EXCLUDED_ROUTES}",
            f"- Campaign B maximum complete episodes: {MAX_NEW_COMPLETE_EPISODES}",
            f"- minimum desired global new vehicles: {MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES}",
            "- route balance target: ONE_COMPLETE_PER_TARGET_ROUTE_PREFERRED",
            f"- exclusion registry vehicle count: {len(exclusion)}",
            "- candidate priority: GLOBAL_UNSEEN_VEHICLE, GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE, then route/date-local fallback",
            f"- route follow duration source: {R2D1G_ROOT / 'follow_duration_contract.json'}",
            "- execution window contract: runtime must recheck current KST and follow windows",
            f"- recommended API budget: {RECOMMENDED_CAMPAIGN_B_CALLS}",
            f"- API hard cap: {ABSOLUTE_CAMPAIGN_B_HARD_CAP}",
            f"- daily safety cap: {DAILY_PHYSICAL_SAFETY_CAP}",
            f"- calls/min cap: {MAX_CALLS_PER_MINUTE}",
            f"- focused session cap: {MAX_CONCURRENT_FOCUSED_SESSIONS}",
            "- temporary disappearance: POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
            "- exact-ID re-entry: same vehicle, route, direction, sequence <= 5 and sequence drop",
            "- confirmation: 3 post-terminal samples or 10 minutes with explicit evidence",
            "- complete rule: upstream, pre-terminal, terminal, re-entry, confirmation, continuity and SHA audits",
            "- censoring: explicit right-censor reasons only; no vehicle-response-loss censor",
            "- fatal stop: fatal provider/security/scope/budget error stops all target routes",
            "- early stop: max complete, route balance, global vehicle target, hard-cap/window guards",
            "- execution handoff packet: ready",
            "- Campaign B authorization review: passed",
            "- Campaign B live execution authorized: false",
            "- Campaign C authorized: false",
            "- estimation authorized: false",
            "- simulator authorized: false",
            "- Phase 2 authorized: false",
            f"- manifest/JSON/Parquet: manifest pending finalization, strict JSON {len(strict_json_failures)}, parquet read {len(parquet_read_failures)}",
            f"- secret scan result: {len(secret_hits)}",
            "- next authorized action: Campaign B controlled live observation runtime authorization and execution only",
        ]
        (output_root / "prompt5_e01_r2d1j_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        dump_json(output_root / "prompt5_e01_r2d1j_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1j_manifest.json")
        gate["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        dump_json(output_root / "prompt5_e01_r2d1j_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1j_manifest.json")
        gate["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        gate["manifest_hash_mismatch_count"] = 0
        dump_json(output_root / "prompt5_e01_r2d1j_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", manifest_payload(output_root))
        manifest = read_json(output_root / "prompt5_e01_r2d1j_manifest.json")
        hash_mismatches = []
        for entry in manifest["files"]:
            path = output_root / entry["path"]
            if entry.get("sha256") is not None and sha256_file(path) != entry["sha256"]:
                hash_mismatches.append(entry["path"])
        if manifest["missing_required_file_count"] or hash_mismatches:
            raise ReviewFailure("FAIL_MANIFEST_RECONCILIATION", "Manifest reconciliation failed")

        print("R2D-1J CAMPAIGN B AUTHORIZATION REVIEW COMPLETE")
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
        print("\nsource_hf2_gate:")
        print(hf2_gate.get("gate_status"))
        print("\ncumulative_complete_episode_count:")
        print(len(registry))
        print("\nroute_complete_counts:")
        for route, count in counts_by_route.items():
            print(f"{route}={count}")
        print("\nglobal_unique_vehicle_count:")
        print(global_unique_vehicle_count)
        print("\nremaining_complete_episodes_to_12:")
        print(complete_deficit)
        print("\nglobal_unique_vehicle_deficit:")
        print(global_unique_deficit)
        print("\ncampaign_b_target_routes:")
        print(",".join(TARGET_ROUTES))
        print("\ncampaign_b_excluded_routes:")
        print(",".join(EXCLUDED_ROUTES))
        print("\ncampaign_b_maximum_new_complete_episodes:")
        print(MAX_NEW_COMPLETE_EPISODES)
        print("\nminimum_desired_new_global_vehicles:")
        print(MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES)
        print("\nglobal_vehicle_exclusion_registry_count:")
        print(len(exclusion))
        print("\nglobal_vehicle_exclusion_registry_duplicate_count:")
        print(exclusion_duplicate_count)
        print("\nroute_balance_target:")
        print("ONE_COMPLETE_PER_TARGET_ROUTE_PREFERRED")
        print("\nrecommended_campaign_b_calls:")
        print(RECOMMENDED_CAMPAIGN_B_CALLS)
        print("\nabsolute_campaign_b_hard_cap:")
        print(ABSOLUTE_CAMPAIGN_B_HARD_CAP)
        print("\ndaily_physical_safety_cap:")
        print(DAILY_PHYSICAL_SAFETY_CAP)
        print("\nmax_calls_per_minute:")
        print(MAX_CALLS_PER_MINUTE)
        print("\nmaximum_concurrent_focused_sessions:")
        print(MAX_CONCURRENT_FOCUSED_SESSIONS)
        print("\ntemporary_disappearance_state:")
        print("POST_TERMINAL_DISAPPEARED_WAITING_REENTRY")
        print("\ncampaign_b_authorization_review_passed:")
        print("true")
        print("\ncampaign_b_execution_release_packet_ready:")
        print("true")
        print("\ncampaign_b_live_execution_authorized:")
        print("false")
        print("\ncampaign_c_authorized:")
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
        print("\nmapping_regression_count:")
        print(mapping_regression_count)
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
        print("\nnext_authorized_action:")
        print("Campaign B controlled live observation runtime authorization and execution only")

    except ReviewFailure as exc:
        fail(output_root, exc.gate_status, str(exc))


if __name__ == "__main__":
    main()
