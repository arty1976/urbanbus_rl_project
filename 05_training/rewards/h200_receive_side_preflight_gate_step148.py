from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STEP_ID = 148
ARTIFACT_VERSION = "h200_receive_side_preflight_gate_step148_v1"

REQUIRED_SOURCE_FILES = [
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
    "05_training/rewards/h200_transfer_package_export_manifest_step145.md",
    "05_training/rewards/h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/validate_h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/test_h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/h200_transfer_package_integrity_verifier_step146.md",
    "05_training/rewards/h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/validate_h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/test_h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/h200_receive_side_transfer_runbook_step147.md",
    "05_training/rewards/h200_receive_side_transfer_runbook_step147.py",
    "05_training/rewards/validate_h200_receive_side_transfer_runbook_step147.py",
    "05_training/rewards/test_h200_receive_side_transfer_runbook_step147.py",
    "step147_h200_receive_side_transfer_runbook.ps1",
]

LOCAL_ONLY_PATTERNS = [
    "artifacts/rewards/**",
    "05_training/rewards/*.latest.json",
    "artifacts/baseline_v1/**",
    "1000005000/**",
    "project_files/**",
]

H200_RECEIVE_SIDE_CHECKLIST = [
    "clone_or_copy_package_to_h200_project_root",
    "verify_git_commit_matches_transfer_manifest",
    "run_step146_integrity_verifier_on_h200",
    "confirm_python_version",
    "confirm_torch_import",
    "confirm_cuda_visible",
    "confirm_gpu_count",
    "confirm_checkpoint_output_root_exists_or_can_be_created",
    "confirm_artifact_output_root_exists_or_can_be_created",
    "confirm_no_latest_json_used_as_source_of_truth",
    "confirm_actual_execution_release_flag_is_false",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(project_root: Path, args: List[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(project_root),
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except Exception as exc:
        return f"GIT_UNAVAILABLE: {exc}"


def collect_git_metadata(project_root: Path) -> Dict[str, Any]:
    return {
        "branch": run_git(project_root, ["branch", "--show-current"]),
        "commit": run_git(project_root, ["rev-parse", "HEAD"]),
        "status_short": run_git(project_root, ["status", "--short"]),
        "remote_origin": run_git(project_root, ["remote", "get-url", "origin"]),
    }


def classify_missing_required(project_root: Path) -> List[str]:
    missing = []
    for rel in REQUIRED_SOURCE_FILES:
        if not (project_root / rel).exists():
            missing.append(rel)
    return missing


def build_file_inventory(project_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = project_root / rel
        row: Dict[str, Any] = {
            "relative_path": rel,
            "exists": path.exists(),
            "size_bytes": None,
            "sha256": None,
        }
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def build_manifest(
    project_root: Path,
    output_root: Path,
    h200_project_root: str,
    require_clean_worktree: bool = False,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    hard_failures: List[str] = []
    warnings: List[str] = []

    if not project_root.exists():
        hard_failures.append(f"project_root_not_found: {project_root}")

    missing_required = classify_missing_required(project_root)
    if missing_required:
        hard_failures.append(
            "missing_required_source_files: " + ", ".join(missing_required)
        )

    git_meta = collect_git_metadata(project_root)
    if str(git_meta.get("commit", "")).startswith("GIT_UNAVAILABLE"):
        warnings.append("git_metadata_unavailable")

    status_short = str(git_meta.get("status_short", ""))
    if status_short and not status_short.startswith("GIT_UNAVAILABLE"):
        if require_clean_worktree:
            hard_failures.append("working_tree_not_clean")
        else:
            warnings.append("working_tree_not_clean_but_not_blocking_for_step148")

    if "\\" in h200_project_root:
        hard_failures.append("h200_project_root_must_be_linux_style_path")

    if not h200_project_root.startswith("/"):
        hard_failures.append("h200_project_root_must_be_absolute_linux_path")

    inventory = build_file_inventory(project_root)

    gate_status = (
        "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW"
        if not hard_failures
        else "BLOCKED"
    )

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "h200_project_root": h200_project_root,
        "gate_status": gate_status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "git_metadata": git_meta,
        "required_source_file_count": len(REQUIRED_SOURCE_FILES),
        "missing_required_source_files": missing_required,
        "file_inventory": inventory,
        "local_only_patterns": LOCAL_ONLY_PATTERNS,
        "receive_side_checklist": H200_RECEIVE_SIDE_CHECKLIST,
        "receive_side_expected_commands": [
            "git status --short",
            "git rev-parse HEAD",
            "python 05_training/rewards/h200_transfer_package_integrity_verifier_step146.py --help",
            "python 05_training/rewards/h200_receive_side_preflight_gate_step148.py --help",
        ],
        "operator_handoff_policy": {
            "source_of_truth": [
                "tracked_source_scripts",
                "tracked_validation_scripts",
                "tracked_markdown_specs",
                "tracked_json_specs",
            ],
            "not_source_of_truth": [
                "latest_json_pointer_files",
                "local_artifacts_recreated_from_scripts",
                "windows_absolute_paths",
                "manual_console_claims",
            ],
        },
        "execution_locks": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "next_step_recommendation": {
            "step": 149,
            "title": "H200 environment preflight result manifest",
            "meaning": (
                "Run this receive-side checklist on the H200 server and save "
                "the actual environment result without releasing training."
            ),
        },
    }

    manifest_path = output_root / "h200_receive_side_preflight_gate_step148_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_receive_side_preflight_gate_step148",
    )
    parser.add_argument(
        "--h200-project-root",
        default="/workspace/urbanbus_rl_project",
    )
    parser.add_argument("--require-clean-worktree", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        h200_project_root=args.h200_project_root,
        require_clean_worktree=bool(args.require_clean_worktree),
    )

    print("[OK] Step 148 H200 receive-side preflight gate completed")
    print(f"[OK] gate_status     : {manifest['gate_status']}")
    print(f"[OK] hard_failures   : {len(manifest['hard_failures'])}")
    print(f"[OK] warnings        : {len(manifest['warnings'])}")
    print(f"[OK] h200_root       : {manifest['h200_project_root']}")
    print(f"[OK] manifest        : {manifest['manifest_path']}")
    print(f"[OK] actual_execution_allowed : {manifest['execution_locks']['actual_execution_allowed']}")
    print(f"[OK] train_allowed            : {manifest['execution_locks']['train_allowed']}")

    if manifest["hard_failures"]:
        for failure in manifest["hard_failures"]:
            print(f"[BLOCKED] {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
