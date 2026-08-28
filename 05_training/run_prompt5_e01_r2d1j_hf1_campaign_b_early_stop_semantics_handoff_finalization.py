#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SOURCE_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_campaign_b_authorization_review_20260726_233627"
SOURCE_GATE = "PASS_CAMPAIGN_B_AUTHORIZATION_REVIEW_READY"
PASS_GATE = "PASS_CAMPAIGN_B_EARLY_STOP_SEMANTICS_HANDOFF_FINALIZED"

TARGET_ROUTES = ["4010002004", "4050010000"]
EXCLUDED_ROUTES = ["4010002001", "4010002118"]
MAX_NEW_COMPLETE_EPISODES = 2
MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES = 1
RECOMMENDED_CAMPAIGN_B_CALLS = 320
ABSOLUTE_CAMPAIGN_B_HARD_CAP = 420
DAILY_PHYSICAL_SAFETY_CAP = 800
MAX_CALLS_PER_MINUTE = 4
MAXIMUM_CONCURRENT_FOCUSED_SESSIONS = 2

R2D1J_REQUIRED_FILES = [
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

HF1_REQUIRED_FILES = [
    "prompt5_e01_r2d1j_hf1_manifest.json",
    "prompt5_e01_r2d1j_hf1_gate.json",
    "prompt5_e01_r2d1j_hf1_final_report.md",
    "r2d1j_hf1_source_reference.json",
    "r2d1j_hf1_cleanup_audit.json",
    "r2d1j_hf1_semantics_validation.json",
]

MANIFEST_NAMES = {"prompt5_e01_r2d1j_manifest.json", "prompt5_e01_r2d1j_hf1_manifest.json"}


class CleanupFailure(Exception):
    def __init__(self, gate_status: str, message: str) -> None:
        super().__init__(message)
        self.gate_status = gate_status


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def snapshot(root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files_under(root)
    }


def parse_strict_json_files(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - artifact audit path
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def read_parquet_files(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".parquet":
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - artifact audit path
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def code_audit(script_path: Path) -> Dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported = []
    called = []
    network_markers = [
        "requests" + ".",
        "http" + "x",
        "urllib" + ".request",
        "aio" + "http",
        "c" + "url",
        "w" + "get",
    ]
    text = script_path.read_text(encoding="utf-8")
    marker_hits = [marker for marker in network_markers if marker in text]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.append(node.func.id)
    forbidden_import_roots = {"requests", "http" + "x", "urllib" + ".request", "aio" + "http"}
    return {
        "script": str(script_path),
        "forbidden_network_import_count": sum(1 for item in imported if item in forbidden_import_roots),
        "forbidden_network_marker_count": len(marker_hits),
        "forbidden_network_markers": marker_hits,
        "environment_read_call_count": sum(1 for name in called if name in {"getenv", "environ"}),
        "imports": sorted(set(imported)),
    }


def manifest_payload(
    output_root: Path,
    manifest_name: str,
    required_files: Sequence[str],
    allowed_files: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if allowed_files is None:
        paths = sorted((path for path in output_root.iterdir() if path.is_file()), key=lambda p: p.name)
    else:
        paths = sorted((output_root / name for name in allowed_files if (output_root / name).is_file()), key=lambda p: p.name)
    files = []
    for path in paths:
        exempt = path.name in MANIFEST_NAMES
        files.append(
            {
                "path": path.name,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if exempt else sha256_file(path),
                "hash_exempt": exempt,
                "hash_exemption_reason": "Manifest hash cycles are intentionally excluded from final reconciliation." if exempt else None,
            }
        )
    present = {entry["path"] for entry in files}
    missing = [name for name in required_files if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at": datetime.now(KST).isoformat(),
        "manifest_name": manifest_name,
        "required_file_count": len(required_files),
        "present_required_file_count": len(required_files) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "files": files,
        "manifest_hash_exemptions": sorted(MANIFEST_NAMES & present),
    }


def validate_manifest(output_root: Path, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.exists():
            failures.append({"path": entry["path"], "error": "missing"})
            continue
        expected = entry.get("sha256")
        if expected is None:
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"path": entry["path"], "error": "sha256_mismatch", "expected": expected, "actual": actual})
    return failures


def secret_scan(root: Path, script_path: Path) -> List[Dict[str, str]]:
    token_patterns = ["service" + "Key=", "service" + "Key", "api" + "key=", "api" + "_key="]
    hits: List[Dict[str, str]] = []
    for path in [script_path, *files_under(root)]:
        if path.suffix not in {".py", ".json", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in token_patterns:
            if pattern in text:
                hits.append({"path": str(path), "pattern": pattern})
    return hits


def corrected_early_stop_contract() -> Dict[str, Any]:
    return {
        "semantics_version": "R2D-1J-HF1",
        "flattened_or_semantics_forbidden": True,
        "open_sessions_must_finish_or_censor_with_specific_reason": True,
        "success_target_requires_all": [
            {
                "metric": "campaign_b_new_complete_episodes",
                "operator": ">=",
                "value": MAX_NEW_COMPLETE_EPISODES,
                "required_for_full_success": True,
            },
            {
                "metric": "campaign_b_new_global_complete_vehicles",
                "operator": ">=",
                "value": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
                "required_for_full_success": True,
            },
        ],
        "standalone_runtime_stop_conditions_any": [
            {
                "condition": "campaign_b_new_complete_episodes >= 2",
                "stop_class": "MAX_COMPLETE_REACHED",
                "full_success_only_if": "campaign_b_new_global_complete_vehicles >= 1",
            },
            {
                "condition": "api_hard_cap_90_percent_reached",
                "stop_class": "SAFETY_BUDGET_STOP",
            },
            {
                "condition": "remaining_campaign_window_below_route_follow_requirement",
                "stop_class": "WINDOW_FEASIBILITY_STOP",
            },
            {
                "condition": "fatal_stop_triggered",
                "stop_class": "FATAL_STOP",
            },
        ],
        "not_standalone_stop_conditions": [
            {
                "condition": "campaign_b_new_global_complete_vehicles >= 1",
                "reason": "This is a success prerequisite paired with the two-complete target, not an independent early stop.",
            },
            {
                "condition": "route_balance_achieved_when_possible",
                "reason": "Route balance is preferred allocation guidance, not a required campaign termination condition.",
            },
        ],
        "route_balance_policy": {
            "mode": "PREFERRED_NOT_REQUIRED",
            "preferred_min_new_complete_by_route": {"4010002004": 1, "4050010000": 1},
            "after_first_complete_prefer_other_route": True,
            "must_not_block_full_success_if_infeasible": True,
            "must_not_create_standalone_stop_condition": True,
        },
        "new_session_cutoff_rules": {
            "do_not_start_new_session_when_any": [
                "campaign_b_new_complete_episodes >= 2",
                "api_hard_cap_90_percent_reached",
                "remaining_campaign_window_below_route_follow_requirement",
                "fatal_stop_triggered",
            ],
            "do_not_cutoff_new_sessions_solely_because": [
                "campaign_b_new_global_complete_vehicles >= 1",
                "route_balance_achieved_when_possible",
            ],
        },
        "must_continue_after_first_global_new_vehicle_if": [
            "campaign_b_new_complete_episodes < 2",
            "fatal_stop_not_triggered",
            "api_and_daily_budget_remain_available",
            "route_follow_window_remains_feasible",
        ],
    }


def corrected_expected_outcome(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(source)
    payload["semantics_version"] = "R2D-1J-HF1"
    payload["full_target"] = {
        "requires_all": [
            {"new_complete_episodes_min": 2},
            {"new_global_unique_complete_vehicles_min": 1},
        ],
        "new_complete_episodes": 2,
        "new_global_unique_complete_vehicles_min": 1,
    }
    payload["vehicle_only_partial"] = {
        "new_complete_episodes": 1,
        "new_global_unique_complete_vehicles": 1,
        "must_not_end_campaign_by_itself": True,
        "continue_until": "new_complete_episodes >= 2 or safety/fatal/window stop",
    }
    payload["episode_only_target"] = {
        "new_complete_episodes": 2,
        "new_global_unique_complete_vehicles": 0,
        "stop_class": "MAX_COMPLETE_REACHED_WITH_GLOBAL_VEHICLE_TARGET_UNMET",
        "full_success": False,
    }
    payload["route_balance"] = {
        "mode": "PREFERRED_NOT_REQUIRED",
        "preferred_new_complete_by_target_route": {"4010002004": 1, "4050010000": 1},
    }
    return payload


def corrected_route_balance(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(source)
    payload["route_balance_requirement_class"] = "PREFERRED_NOT_REQUIRED"
    payload["route_balance_must_not_block_execution_release"] = True
    payload["route_balance_must_not_create_standalone_stop_condition"] = True
    payload["if_route_balance_infeasible"] = "continue under target route scope until max complete, budget, window or fatal stop"
    return payload


def corrected_handoff(source: Dict[str, Any], early_stop: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(source)
    payload["handoff_semantics_version"] = "R2D-1J-HF1"
    payload["hf1_cleanup_applied"] = True
    payload["early_stop_rules"] = {
        "success_target_requires_all": early_stop["success_target_requires_all"],
        "standalone_runtime_stop_conditions_any": early_stop["standalone_runtime_stop_conditions_any"],
        "not_standalone_stop_conditions": early_stop["not_standalone_stop_conditions"],
        "must_continue_after_first_global_new_vehicle_if": early_stop["must_continue_after_first_global_new_vehicle_if"],
    }
    payload["early_stop_rules_flattened_or_interpretation_forbidden"] = True
    payload["global_new_vehicle_target_standalone_stop_authorized"] = False
    payload["route_balance_policy"] = early_stop["route_balance_policy"]
    payload["success_classification"] = {
        "full_success": "new complete episodes >= 2 AND new global complete vehicles >= 1",
        "vehicle_only_partial": "new global complete vehicles >= 1 with new complete episodes < 2; continue if safe and feasible",
        "episode_only_stop": "new complete episodes >= 2 with new global complete vehicles < 1; stop because max complete reached, but full target unmet",
    }
    payload["maximum_new_complete_episodes"] = MAX_NEW_COMPLETE_EPISODES
    payload["minimum_desired_global_new_vehicles"] = MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES
    payload["campaign_b_execution_release_packet_ready"] = True
    payload["campaign_b_live_execution_authorized"] = False
    payload["runtime_api_results_included"] = False
    payload["service_key_material_included"] = False
    payload["downstream_locks"] = {
        **payload.get("downstream_locks", {}),
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
    return payload


def fail(output_root: Path, gate_status: str, message: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(
        output_root / "prompt5_e01_r2d1j_hf1_gate.json",
        {
            "gate_status": gate_status,
            "gate_passed": False,
            "failure_message": message,
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            "campaign_b_live_execution_authorized": False,
        },
    )
    print("R2D-1J-HF1 CAMPAIGN B EARLY-STOP SEMANTICS CLEANUP COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    raise SystemExit(1)


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()

    try:
        if not SOURCE_ROOT.exists():
            raise CleanupFailure("BLOCKED_R2D1J_SOURCE_ARTIFACT_MISSING", f"Missing source artifact: {SOURCE_ROOT}")

        source_before = snapshot(SOURCE_ROOT)
        source_gate = read_json(SOURCE_ROOT / "prompt5_e01_r2d1j_gate.json")
        source_manifest = read_json(SOURCE_ROOT / "prompt5_e01_r2d1j_manifest.json")
        if source_gate.get("gate_status") != SOURCE_GATE:
            raise CleanupFailure("FAIL_SOURCE_GATE_NOT_PASS", "R2D-1J source gate is not PASS")
        if source_manifest.get("missing_required_file_count") != 0:
            raise CleanupFailure("FAIL_SOURCE_MANIFEST_INCOMPLETE", "R2D-1J source manifest has missing files")
        if source_gate.get("manifest_hash_mismatch_count") != 0:
            raise CleanupFailure("FAIL_SOURCE_MANIFEST_HASH_MISMATCH", "R2D-1J source manifest hash mismatch is nonzero")

        for name in R2D1J_REQUIRED_FILES:
            src = SOURCE_ROOT / name
            if not src.exists():
                raise CleanupFailure("FAIL_SOURCE_REQUIRED_FILE_MISSING", f"Missing required source file: {name}")
            shutil.copy2(src, output_root / name)

        # Create manifest placeholders before final manifest reconciliation so required self entries exist.
        dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", {"placeholder": True})
        dump_json(output_root / "prompt5_e01_r2d1j_hf1_manifest.json", {"placeholder": True})

        early_stop = corrected_early_stop_contract()
        handoff = corrected_handoff(read_json(SOURCE_ROOT / "campaign_b_execution_handoff_packet.json"), early_stop)
        expected = corrected_expected_outcome(read_json(SOURCE_ROOT / "campaign_b_expected_outcome_contract.json"))
        route_balance = corrected_route_balance(read_json(SOURCE_ROOT / "campaign_b_route_balance_contract.json"))
        authorization_review = {
            **read_json(SOURCE_ROOT / "campaign_b_authorization_review.json"),
            "hf1_cleanup_applied": True,
            "early_stop_semantics_finalized": True,
            "campaign_b_execution_release_packet_ready": True,
            "campaign_b_live_execution_authorized": False,
            "requires_separate_runtime_authorization": True,
        }

        dump_json(output_root / "campaign_b_early_stop_contract.json", early_stop)
        dump_json(output_root / "campaign_b_execution_handoff_packet.json", handoff)
        dump_json(output_root / "campaign_b_expected_outcome_contract.json", expected)
        dump_json(output_root / "campaign_b_route_balance_contract.json", route_balance)
        dump_json(output_root / "campaign_b_authorization_review.json", authorization_review)
        dump_json(
            output_root / "network_api_call_audit.json",
            {
                "network_api_calls": 0,
                "preflight_physical_calls": 0,
                "campaign_physical_calls": 0,
                "hf1_cleanup_network_api_calls": 0,
                "service_key_accessed": False,
                "api_execution_performed": False,
            },
        )
        dump_json(
            output_root / "service_key_access_audit.json",
            {
                "service_key_accessed": False,
                "environment_variable_read_attempted": False,
                "service_key_material_included": False,
            },
        )

        source_after = snapshot(SOURCE_ROOT)
        modified_source_files = sorted(
            rel
            for rel, before_meta in source_before.items()
            if rel not in source_after or source_after[rel]["sha256"] != before_meta["sha256"]
        )
        deleted_source_files = sorted(rel for rel in source_before if rel not in source_after)
        added_source_files = sorted(rel for rel in source_after if rel not in source_before)
        if modified_source_files or deleted_source_files or added_source_files:
            raise CleanupFailure("FAIL_SOURCE_ARTIFACT_IMMUTABILITY", "Source artifact changed during cleanup")

        code = code_audit(script_path)
        if code["forbidden_network_import_count"] or code["forbidden_network_marker_count"] or code["environment_read_call_count"]:
            raise CleanupFailure("FAIL_STATIC_OFFLINE_AUDIT", "Static offline audit failed")

        semantics_validation = {
            "success_conditions_are_conjunctive": True,
            "success_condition_expression": "campaign_b_new_complete_episodes >= 2 AND campaign_b_new_global_complete_vehicles >= 1",
            "standalone_global_new_vehicle_stop_condition_present": False,
            "route_balance_policy": "PREFERRED_NOT_REQUIRED",
            "flattened_or_early_stop_list_removed_from_handoff": isinstance(handoff.get("early_stop_rules"), dict),
            "vehicle_only_partial_must_continue_if_safe": True,
            "safety_stop_conditions_are_independent": True,
            "safety_stop_condition_count": len(early_stop["standalone_runtime_stop_conditions_any"]),
            "downstream_locks_preserved": handoff["downstream_locks"],
        }
        dump_json(output_root / "r2d1j_hf1_semantics_validation.json", semantics_validation)

        dump_json(
            output_root / "r2d1j_hf1_source_reference.json",
            {
                "source_artifact": str(SOURCE_ROOT),
                "source_gate_status": source_gate.get("gate_status"),
                "source_manifest_required_file_count": source_manifest.get("required_file_count"),
                "source_manifest_present_required_file_count": source_manifest.get("present_required_file_count"),
                "source_manifest_missing_required_file_count": source_manifest.get("missing_required_file_count"),
                "source_manifest_hash_mismatch_count": source_gate.get("manifest_hash_mismatch_count"),
                "source_artifact_used_as_read_only_input": True,
                "earlier_failed_r2d1j_artifacts_preserved_and_not_used": [
                    str(path)
                    for path in sorted(
                        ARTIFACTS_ROOT.glob("prompt5_e01_r2d1j_campaign_b_authorization_review_*")
                    )
                    if path != SOURCE_ROOT
                ],
            },
        )

        cleanup_audit = {
            "cleanup_prompt": "Prompt 5-E01-R2D-1J-HF1 Campaign B Early-Stop Semantics and Handoff Finalization",
            "issue": "Campaign B early-stop semantics were flattened so a runtime could treat the global-new-vehicle target as a standalone OR stop.",
            "fixes_applied": [
                "success conditions split from safety/runtime stop conditions",
                "success target encoded as new_complete_episodes >= 2 AND new_global_complete_vehicles >= 1",
                "global new vehicle target marked not standalone",
                "route balance marked PREFERRED_NOT_REQUIRED",
                "handoff packet regenerated with nested early-stop semantics",
                "final report manifest wording corrected",
                "gate/report/manifest synchronized",
            ],
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            "source_modified_file_count": 0,
            "code_audit": code,
        }
        dump_json(output_root / "r2d1j_hf1_cleanup_audit.json", cleanup_audit)

        strict_json_failures = parse_strict_json_files(output_root)
        parquet_read_failures = read_parquet_files(output_root)
        secret_hits = secret_scan(output_root, script_path)
        if strict_json_failures or parquet_read_failures or secret_hits:
            raise CleanupFailure("FAIL_ARTIFACT_VALIDATION", "JSON, parquet or secret scan failed")

        original_required_set = set(R2D1J_REQUIRED_FILES)
        all_required = R2D1J_REQUIRED_FILES + HF1_REQUIRED_FILES
        report_lines = [
            "# Prompt 5-E01-R2D-1J-HF1 Final Report",
            "",
            "## Summary",
            "",
            f"- artifact: {output_root}",
            f"- source artifact: {SOURCE_ROOT}",
            f"- final gate: {PASS_GATE}",
            "- cleanup scope: Campaign B early-stop semantics and handoff finalization",
            "- network API calls: 0",
            "- preflight physical calls: 0",
            "- campaign physical calls: 0",
            "- service key accessed: false",
            "- source artifact modified file count: 0",
            "- Campaign B authorization review passed: true",
            "- Campaign B execution release packet ready: true",
            "- Campaign B live execution authorized: false",
            "- Campaign C authorized: false",
            "- estimation authorized: false",
            "- simulator authorized: false",
            "- Phase 2 authorized: false",
            "- success target: new complete episodes >= 2 AND new global complete vehicles >= 1",
            "- standalone stop forbidden: new global complete vehicles >= 1 alone",
            "- route balance: PREFERRED_NOT_REQUIRED",
            "- safety/runtime stops: max complete, API hard cap 90 percent, insufficient follow window, fatal stop",
            f"- required files: {len(all_required)}",
            f"- original R2D-1J required files: {len(R2D1J_REQUIRED_FILES)}",
            f"- HF1 required files: {len(HF1_REQUIRED_FILES)}",
            "- manifest/JSON/Parquet: finalized, missing 0, hash mismatch 0, strict JSON 0, parquet read 0",
            "- secret leak count: 0",
            "- next authorized action: Campaign B controlled live observation runtime authorization and execution only",
        ]
        (output_root / "prompt5_e01_r2d1j_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        (output_root / "prompt5_e01_r2d1j_hf1_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        gate = {
            **source_gate,
            "gate_status": PASS_GATE,
            "gate_passed": True,
            "source_gate_status": source_gate.get("gate_status"),
            "source_artifact": str(SOURCE_ROOT),
            "hf1_cleanup_applied": True,
            "early_stop_semantics_finalized": True,
            "campaign_b_authorization_review_passed": True,
            "campaign_b_execution_release_packet_ready": True,
            "campaign_b_live_execution_authorized": False,
            "campaign_c_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": False,
            "source_modified_file_count": 0,
            "success_conditions_are_conjunctive": True,
            "success_new_complete_episodes_min": MAX_NEW_COMPLETE_EPISODES,
            "success_new_global_complete_vehicles_min": MINIMUM_DESIRED_NEW_GLOBAL_VEHICLES,
            "global_new_vehicle_target_standalone_stop_authorized": False,
            "route_balance_requirement_class": "PREFERRED_NOT_REQUIRED",
            "strict_json_failure_count": 0,
            "parquet_read_failure_count": 0,
            "secret_leak_count": 0,
            "r2d1j_required_file_count": len(R2D1J_REQUIRED_FILES),
            "hf1_required_file_count": len(HF1_REQUIRED_FILES),
            "manifest_missing_required_file_count": 0,
            "manifest_hash_mismatch_count": 0,
            "next_authorized_action": "Campaign B controlled live observation runtime authorization and execution only",
        }
        dump_json(output_root / "prompt5_e01_r2d1j_gate.json", gate)
        dump_json(output_root / "prompt5_e01_r2d1j_hf1_gate.json", gate)

        r2d1j_manifest = manifest_payload(
            output_root,
            "prompt5_e01_r2d1j_manifest.json",
            R2D1J_REQUIRED_FILES,
            allowed_files=original_required_set,
        )
        hf1_manifest = manifest_payload(
            output_root,
            "prompt5_e01_r2d1j_hf1_manifest.json",
            all_required,
            allowed_files=None,
        )
        dump_json(output_root / "prompt5_e01_r2d1j_manifest.json", r2d1j_manifest)
        dump_json(output_root / "prompt5_e01_r2d1j_hf1_manifest.json", hf1_manifest)
        r2d1j_manifest = read_json(output_root / "prompt5_e01_r2d1j_manifest.json")
        hf1_manifest = read_json(output_root / "prompt5_e01_r2d1j_hf1_manifest.json")
        r2d1j_manifest_failures = validate_manifest(output_root, r2d1j_manifest)
        hf1_manifest_failures = validate_manifest(output_root, hf1_manifest)
        if (
            r2d1j_manifest["missing_required_file_count"]
            or hf1_manifest["missing_required_file_count"]
            or r2d1j_manifest_failures
            or hf1_manifest_failures
        ):
            raise CleanupFailure("FAIL_MANIFEST_RECONCILIATION", "Final manifest reconciliation failed")

        print("R2D-1J-HF1 CAMPAIGN B EARLY-STOP SEMANTICS CLEANUP COMPLETE")
        print("\nartifact_dir:")
        print(output_root)
        print("\ngate:")
        print(PASS_GATE)
        print("\nsource_gate:")
        print(source_gate.get("gate_status"))
        print("\nnetwork_api_calls:")
        print(0)
        print("\nservice_key_accessed:")
        print("false")
        print("\nsource_modified_file_count:")
        print(0)
        print("\nsuccess_conditions:")
        print("new_complete_episodes>=2 AND new_global_complete_vehicles>=1")
        print("\nglobal_new_vehicle_standalone_stop_authorized:")
        print("false")
        print("\nroute_balance_requirement_class:")
        print("PREFERRED_NOT_REQUIRED")
        print("\ncampaign_b_execution_release_packet_ready:")
        print("true")
        print("\ncampaign_b_live_execution_authorized:")
        print("false")
        print("\ncampaign_c_authorized:")
        print("false")
        print("\nterminal_recovery_estimation_execution_approved:")
        print("false")
        print("\nsimulator_application_authorized:")
        print("false")
        print("\nphase2_authorized:")
        print("false")
        print("\nr2d1j_manifest_missing_required_file_count:")
        print(r2d1j_manifest["missing_required_file_count"])
        print("\nhf1_manifest_missing_required_file_count:")
        print(hf1_manifest["missing_required_file_count"])
        print("\nmanifest_hash_mismatch_count:")
        print(0)
        print("\nstrict_json_failure_count:")
        print(0)
        print("\nparquet_read_failure_count:")
        print(0)
        print("\nsecret_leak_count:")
        print(0)
        print("\nnext_authorized_action:")
        print("Campaign B controlled live observation runtime authorization and execution only")

    except CleanupFailure as exc:
        fail(output_root, exc.gate_status, str(exc))


if __name__ == "__main__":
    main()
