$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = Join-Path $ProjectRoot "05_training\rewards"
if (-not (Test-Path $RewardsDir)) {
    New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null
}

$PyMain = Join-Path $RewardsDir "h200_execution_package_boundary_manifest_step144.py"
$PyValidate = Join-Path $RewardsDir "validate_h200_execution_package_boundary_manifest_step144.py"
$PyTest = Join-Path $RewardsDir "test_h200_execution_package_boundary_manifest_step144.py"
$MdDoc = Join-Path $RewardsDir "h200_execution_package_boundary_manifest_step144.md"

$MainCode = @'
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 144
ARTIFACT_VERSION = "h200_execution_package_boundary_manifest_step144_v1"

STEP143_POINTER = "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.latest.json"
STEP143_FALLBACK = "artifacts/rewards/a_family_72run_release_matrix_extension_draft_step143/a_family_72run_release_matrix_extension_draft_step143_manifest.json"

CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

H200_TRANSFER_CANDIDATES = [
    "05_training/rewards/final_reward_spec_step111.md",
    "05_training/rewards/final_reward_spec_step111.json",
    "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.md",
    "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/validate_a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/test_a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/h200_execution_package_boundary_manifest_step144.md",
    "05_training/rewards/h200_execution_package_boundary_manifest_step144.py",
    "05_training/rewards/validate_h200_execution_package_boundary_manifest_step144.py",
    "05_training/rewards/test_h200_execution_package_boundary_manifest_step144.py",
]

LOCAL_ONLY_PATTERNS = [
    "artifacts/rewards/**",
    "05_training/rewards/*.latest.json",
    "artifacts/baseline_v1/**",
    "1000005000/**",
    "project_files/**",
]

REQUIRED_BEFORE_H200_EXECUTION = [
    "explicit_operator_release_manifest",
    "clean_git_status_or_recorded_dirty_diff",
    "pinned_git_commit_hash",
    "h200_environment_report",
    "checkpoint_output_contract",
    "empty_or_archived_result_output_root",
    "no_latest_pointer_as_source_of_truth",
    "no_paper_level_claim_before_result_ingestion",
    "no_winner_selection_before_actual_result_ingestion",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = project_root / p
    return p


def find_step143_manifest(project_root: Path) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP143_POINTER

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            raw = payload.get("manifest_path")
            if raw:
                p = resolve_path(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append(f"step143_pointer_manifest_missing: {p}")
            else:
                warnings.append("step143_pointer_has_no_manifest_path")
        except Exception as exc:
            warnings.append(f"step143_pointer_unreadable: {exc}")
    else:
        warnings.append(f"step143_pointer_missing: {STEP143_POINTER}")

    fallback = project_root / STEP143_FALLBACK
    if fallback.exists():
        warnings.append(f"using_step143_fallback_manifest: {STEP143_FALLBACK}")
        return fallback, warnings

    return None, warnings


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []
    ignored_token_presence_maps = {"required_source_tokens"}

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in ignored_token_presence_maps:
                continue
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    return violations


def inspect_step143(project_root: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    violations: List[str] = []

    manifest_path, find_warnings = find_step143_manifest(project_root)
    warnings.extend(find_warnings)

    result: Dict[str, Any] = {
        "manifest_path": str(manifest_path) if manifest_path else "",
        "draft_path": "",
        "exists": bool(manifest_path and manifest_path.exists()),
        "warnings": warnings,
        "violations": violations,
        "manifest_summary": {},
        "draft_summary": {},
    }

    if manifest_path is None or not manifest_path.exists():
        violations.append("step143_manifest_missing")
        return result

    try:
        manifest = load_json_any(manifest_path)
    except Exception as exc:
        violations.append(f"step143_manifest_unreadable: {exc}")
        return result

    result["manifest_summary"] = {
        "step": manifest.get("step"),
        "artifact_version": manifest.get("artifact_version"),
        "audit_status": manifest.get("audit_status"),
        "matrix_status": manifest.get("matrix_status"),
        "planned_run_count": manifest.get("planned_run_count"),
        "conditions": manifest.get("conditions"),
        "reward_ids": manifest.get("reward_ids"),
        "seeds": manifest.get("seeds"),
    }

    if manifest.get("step") != 143:
        violations.append(f"step143_step_mismatch: {manifest.get('step')}")
    if manifest.get("artifact_version") != "a_family_72run_release_matrix_extension_draft_step143_v1":
        violations.append(f"step143_artifact_version_mismatch: {manifest.get('artifact_version')}")
    if manifest.get("audit_status") != "PASS":
        violations.append(f"step143_audit_not_pass: {manifest.get('audit_status')}")
    if manifest.get("matrix_status") != "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_READY_ACTUAL_STILL_LOCKED":
        violations.append(f"step143_matrix_status_invalid: {manifest.get('matrix_status')}")
    if int(manifest.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        violations.append(f"step143_planned_run_count_not_72: {manifest.get('planned_run_count')}")
    if manifest.get("conditions") != CONDITIONS:
        violations.append(f"step143_conditions_mismatch: {manifest.get('conditions')}")
    if manifest.get("reward_ids") != REWARD_IDS:
        violations.append(f"step143_reward_ids_mismatch: {manifest.get('reward_ids')}")
    if manifest.get("seeds") != SEEDS:
        violations.append(f"step143_seeds_mismatch: {manifest.get('seeds')}")

    violations.extend([f"step143_manifest.{v}" for v in recursive_forbidden_true(manifest)])

    raw_draft = manifest.get("draft_path")
    if not raw_draft:
        violations.append("step143_draft_path_missing")
        return result

    draft_path = resolve_path(project_root, str(raw_draft))
    result["draft_path"] = str(draft_path)

    if not draft_path.exists():
        violations.append(f"step143_draft_missing: {draft_path}")
        return result

    try:
        draft = load_json_any(draft_path)
    except Exception as exc:
        violations.append(f"step143_draft_unreadable: {exc}")
        return result

    run_matrix = draft.get("run_matrix", [])
    condition_counts = dict(Counter(str(r.get("condition_id")) for r in run_matrix))
    reward_counts = dict(Counter(str(r.get("reward_id")) for r in run_matrix))
    seed_counts = {str(k): v for k, v in Counter(int(r.get("seed")) for r in run_matrix).items()} if run_matrix else {}

    result["draft_summary"] = {
        "release_matrix_version": draft.get("release_matrix_version"),
        "created_from_step": draft.get("created_from_step"),
        "planned_run_count": draft.get("planned_run_count"),
        "run_matrix_count": len(run_matrix),
        "condition_counts": condition_counts,
        "reward_counts": reward_counts,
        "seed_counts": seed_counts,
        "source_steps": draft.get("source_steps", {}),
        "baseline_reference": draft.get("baseline_reference", {}),
        "release_decision": draft.get("release_decision", {}),
    }

    if draft.get("release_matrix_version") != "a_family_72run_release_matrix_v1":
        violations.append(f"draft_release_matrix_version_mismatch: {draft.get('release_matrix_version')}")
    if draft.get("created_from_step") != 143:
        violations.append(f"draft_created_from_step_mismatch: {draft.get('created_from_step')}")
    if int(draft.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        violations.append(f"draft_planned_run_count_not_72: {draft.get('planned_run_count')}")
    if len(run_matrix) != EXPECTED_RUN_COUNT:
        violations.append(f"draft_run_matrix_count_not_72: {len(run_matrix)}")

    run_ids = [str(r.get("run_id")) for r in run_matrix]
    if len(set(run_ids)) != len(run_ids):
        violations.append("draft_run_id_duplicate")

    for condition in CONDITIONS:
        if condition_counts.get(condition, 0) != 18:
            violations.append(f"draft_condition_count_mismatch: {condition}={condition_counts.get(condition, 0)}")
    for reward_id in REWARD_IDS:
        if reward_counts.get(reward_id, 0) != 12:
            violations.append(f"draft_reward_count_mismatch: {reward_id}={reward_counts.get(reward_id, 0)}")
    for seed in SEEDS:
        if seed_counts.get(str(seed), 0) != 24:
            violations.append(f"draft_seed_count_mismatch: {seed}={seed_counts.get(str(seed), 0)}")

    decision = draft.get("release_decision", {})
    if bool(decision.get("actual_execution_allowed", False)):
        violations.append("draft_actual_execution_allowed_true")
    if bool(decision.get("actual_execution_released", False)):
        violations.append("draft_actual_execution_released_true")
    if bool(decision.get("release_ready", False)):
        violations.append("draft_release_ready_true")

    violations.extend([f"step143_draft.{v}" for v in recursive_forbidden_true(draft)])

    return result


def build_h200_package_boundary(project_root: Path, step143_check: Dict[str, Any]) -> Dict[str, Any]:
    transfer_candidates: List[Dict[str, Any]] = []
    for rel in H200_TRANSFER_CANDIDATES:
        p = project_root / rel
        transfer_candidates.append({
            "path": rel,
            "exists": p.exists(),
            "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            "role": "source_or_contract_for_h200_transfer",
        })

    missing_transfer_candidates = [
        item["path"] for item in transfer_candidates if not item["exists"]
    ]

    return {
        "package_boundary_version": "h200_execution_package_boundary_v1",
        "created_from_step": 144,
        "purpose": "Boundary manifest only. This file does not release actual execution.",
        "matrix_source": {
            "source_step": 143,
            "manifest_path": step143_check.get("manifest_path", ""),
            "draft_path": step143_check.get("draft_path", ""),
            "planned_run_count": EXPECTED_RUN_COUNT,
            "conditions": CONDITIONS,
            "reward_ids": REWARD_IDS,
            "seeds": SEEDS,
        },
        "h200_transfer_candidates": transfer_candidates,
        "missing_transfer_candidates": missing_transfer_candidates,
        "local_only_patterns": LOCAL_ONLY_PATTERNS,
        "required_before_h200_execution": REQUIRED_BEFORE_H200_EXECUTION,
        "execution_boundary": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "release_ready": False,
            "release_blocking_reason": "step144_is_package_boundary_manifest_not_operator_release",
        },
        "post_h200_expected_outputs": {
            "per_run": [
                "run_manifest.json",
                "raw_events.parquet",
                "window_rollup.parquet",
                "policy_metadata.json",
                "checkpoint_reference.json",
            ],
            "after_aggregation": [
                "kpi_by_window.parquet",
                "kpi_by_seed.parquet",
                "kpi_by_time_band.parquet",
                "kpi_overall.json",
                "aggregation_manifest.json",
            ],
            "after_ingestion": [
                "actual_reward_ablation_result_schema",
                "selection_criteria_gate",
                "promotion_decision_package",
            ],
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 144 H200 execution package boundary manifest",
        "",
        "This step records the boundary of what should be transferred or referenced for H200 execution preparation.",
        "",
        "It does not execute the 72-run matrix and does not release actual execution.",
        "",
        "## Matrix source",
        "",
        "- source: Step 143 A-family 72-run release matrix draft",
        "- conditions: `A`, `A90`, `A80`, `A70`",
        "- reward_ids: `R0` through `R5`",
        "- seeds: `1`, `2`, `3`",
        "- planned_run_count: `72`",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- package_status: `{payload['package_status']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Boundary",
        "",
        "The Step 143 matrix draft is the source of truth for H200 planning. Local runtime artifacts and latest pointer files are not source-of-truth transfer targets.",
        "",
    ])


def run_generator(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step143_check = inspect_step143(project_root)
    boundary = build_h200_package_boundary(project_root, step143_check)

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    warnings.extend(step143_check.get("warnings", []))
    blocking_reasons.extend(step143_check.get("violations", []))

    # Missing Step 144 files are expected before this script writes them, so do not block
    # on missing h200_execution_package_boundary_manifest_step144.* entries.
    missing_required = [
        p for p in boundary["missing_transfer_candidates"]
        if "h200_execution_package_boundary_manifest_step144" not in p
    ]
    if missing_required:
        warnings.append(f"missing_optional_transfer_candidates: {missing_required}")

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    package_status = (
        "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "H200_EXECUTION_PACKAGE_BOUNDARY_BLOCKED"
    )

    boundary_path = output_root / "h200_execution_package_boundary_step144.json"
    manifest_path = output_root / "h200_execution_package_boundary_manifest_step144.json"
    md_path = output_root / "h200_execution_package_boundary_manifest_step144.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "package_status": package_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "conditions": CONDITIONS,
        "reward_ids": REWARD_IDS,
        "seeds": SEEDS,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step143_check": step143_check,
        "boundary_path": str(boundary_path),
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "scope_note": (
            "Step 144 defines the H200 package boundary. "
            "It does not execute H200 jobs or unlock actual reward ablation."
        ),
        "next_step_recommendation": (
            "Step 145 should create a H200 transfer checklist or export manifest using this boundary."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(boundary_path, boundary)
    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "h200_execution_package_boundary_manifest_step144.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "h200_execution_package_boundary_manifest_step144.latest.json", {
        "manifest_path": str(manifest_path),
        "boundary_path": str(boundary_path),
        "audit_status": audit_status,
        "package_status": package_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/h200_execution_package_boundary_manifest_step144")
    args = parser.parse_args()

    payload = run_generator(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 144 H200 execution package boundary manifest completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] package_status            : {payload['package_status']}")
    print(f"[OK] planned_run_count         : {payload['planned_run_count']}")
    print(f"[OK] conditions                : {payload['conditions']}")
    print(f"[OK] reward_ids                : {payload['reward_ids']}")
    print(f"[OK] seeds                     : {payload['seeds']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] boundary                  : {payload['boundary_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 144 boundary manifest failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@

$ValidateCode = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "h200_execution_package_boundary_manifest_step144_v1"
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                errors.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    return errors


def validate_boundary(boundary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if boundary.get("package_boundary_version") != "h200_execution_package_boundary_v1":
        errors.append("boundary_version_mismatch")
    if boundary.get("created_from_step") != 144:
        errors.append("boundary_created_from_step_must_be_144")

    matrix = boundary.get("matrix_source", {})
    if int(matrix.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("boundary_matrix_planned_run_count_must_be_72")
    if matrix.get("conditions") != CONDITIONS:
        errors.append("boundary_matrix_conditions_mismatch")
    if matrix.get("reward_ids") != REWARD_IDS:
        errors.append("boundary_matrix_reward_ids_mismatch")
    if matrix.get("seeds") != SEEDS:
        errors.append("boundary_matrix_seeds_mismatch")

    decision = boundary.get("execution_boundary", {})
    if bool(decision.get("actual_execution_allowed", False)):
        errors.append("boundary_actual_execution_allowed_true")
    if bool(decision.get("actual_execution_released", False)):
        errors.append("boundary_actual_execution_released_true")
    if bool(decision.get("release_ready", False)):
        errors.append("boundary_release_ready_true")

    guards = boundary.get("non_claim_guards", {})
    if not isinstance(guards, dict) or not guards:
        errors.append("boundary_non_claim_guards_missing")
    else:
        for key, value in guards.items():
            if bool(value):
                errors.append(f"boundary_non_claim_guard_true: {key}")

    return errors


def validate_payload(payload: Dict[str, Any], boundary: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 144:
        errors.append("step_must_be_144")

    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    if payload.get("package_status") not in {
        "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED",
        "H200_EXECUTION_PACKAGE_BOUNDARY_BLOCKED",
    }:
        errors.append("invalid_package_status")
    if require_pass and payload.get("package_status") != "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED":
        errors.append("package_status_not_ready_locked")

    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("payload_planned_run_count_must_be_72")
    if payload.get("conditions") != CONDITIONS:
        errors.append("payload_conditions_mismatch")
    if payload.get("reward_ids") != REWARD_IDS:
        errors.append("payload_reward_ids_mismatch")
    if payload.get("seeds") != SEEDS:
        errors.append("payload_seeds_mismatch")

    errors.extend(recursive_forbidden_true(payload))
    errors.extend(validate_boundary(boundary))

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    boundary = load_json(Path(args.boundary))
    result = validate_payload(payload, boundary, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 144 H200 execution package boundary manifest validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

from validate_h200_execution_package_boundary_manifest_step144 import validate_payload


def base_payload_and_boundary() -> tuple[dict, dict]:
    payload = {
        "artifact_version": "h200_execution_package_boundary_manifest_step144_v1",
        "step": 144,
        "audit_status": "PASS",
        "package_status": "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED",
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }
    boundary = {
        "package_boundary_version": "h200_execution_package_boundary_v1",
        "created_from_step": 144,
        "matrix_source": {
            "planned_run_count": 72,
            "conditions": ["A", "A90", "A80", "A70"],
            "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "seeds": [1, 2, 3],
        },
        "execution_boundary": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "release_ready": False,
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }
    return payload, boundary


def test_pass_payload_and_boundary() -> None:
    payload, boundary = base_payload_and_boundary()
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_actual_execution_true_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_wrong_run_count_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    boundary["matrix_source"]["planned_run_count"] = 18
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("planned_run_count" in e for e in result["errors"])


def test_release_ready_true_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    boundary["execution_boundary"]["release_ready"] = True
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("release_ready" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload, boundary = base_payload_and_boundary()
    payload["audit_status"] = "BLOCKED"
    payload["package_status"] = "H200_EXECUTION_PACKAGE_BOUNDARY_BLOCKED"
    result = validate_payload(payload, boundary, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload_and_boundary()
    test_actual_execution_true_fails()
    test_wrong_run_count_fails()
    test_release_ready_true_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 144 H200 execution package boundary manifest self-test PASS")


if __name__ == "__main__":
    main()
'@

$DocText = @'
# Step 144 H200 execution package boundary manifest

Step 144 defines the boundary for preparing the A-family 72-run matrix for H200 execution.

Source of truth:
- Step 143 A-family 72-run release matrix draft
- A/A90/A80/A70 × R0~R5 × seeds 1,2,3 = 72 runs

This step does not execute H200 jobs.
This step does not release actual execution.
This step does not select a reward winner.
This step does not allow paper-level claims.

The package boundary separates:
- H200 transfer candidates
- local-only runtime artifacts
- required checks before actual H200 execution
- expected outputs after future H200 execution
'@

Set-Content -Path $PyMain -Value $MainCode -Encoding UTF8
Set-Content -Path $PyValidate -Value $ValidateCode -Encoding UTF8
Set-Content -Path $PyTest -Value $TestCode -Encoding UTF8
Set-Content -Path $MdDoc -Value $DocText -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = "artifacts/rewards/h200_execution_package_boundary_manifest_step144"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\h200_execution_package_boundary_manifest_step144.json"
$Boundary = Join-Path $ProjectRoot "$OutputRoot\h200_execution_package_boundary_step144.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 144 H200 execution package boundary manifest failed"
}

& $py $PyValidate --manifest $Manifest --boundary $Boundary
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 144 manifest/boundary validation failed"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 144 self-test failed"
}

Write-Host "[DONE] Step 144 H200 execution package boundary manifest complete."
