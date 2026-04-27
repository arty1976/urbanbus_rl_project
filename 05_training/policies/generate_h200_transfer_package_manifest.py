from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class TransferManifestError(RuntimeError):
    pass


REQUIRED_SOURCE_FILES = [
    "05_training/train_mappo_actual.py",
    "05_training/mappo_runner.py",
    "05_training/simulator_adapter_interface.py",
    "05_training/adapters/historical_replay_adapter.py",
    "05_training/policies/validate_mappo_checkpoint.py",
    "05_training/policies/h200_actual_checkpoint_preflight_checklist.py",
    "05_training/policies/a_family_mappo_actual_execution_plan.py",
    "05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py",
    "05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py",
    "05_training/policies/generate_h200_environment_report.py",
    "05_training/policies/validate_h200_actual_checkpoint_folder.py",
    "05_training/policies/run_h200_training_local_validation_bundle.py",
    "05_training/policies/generate_h200_transfer_package_manifest.py",
    "05_training/policies/H200_server_transfer_checklist.md",
    "05_training/policies/H200_mappo_training_runbook.md",
    "05_training/policies/H200_actual_checkpoint_folder_contract.md",
    "05_training/policies/H200_training_local_validation_bundle.md",
    "05_training/policies/MAPPO_actual_reproducibility_note.md",
]

REQUIRED_ARTIFACTS = [
    "artifacts/baseline_v1/baseline_contract.json",
    "artifacts/experiment_A_v1/experiment_A_contract.json",
    "artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    "artifacts/baseline_v1/B2_rulebased/scenario_index.parquet",
]

CONDITIONAL_GATV2_ARTIFACTS = [
    "artifacts/gatv2_v1/best_mappo.pt",
    "artifacts/gatv2_v1/artifact_contract.json",
    "artifacts/gatv2_v1/snapshot_index.parquet",
    "artifacts/gatv2_v1/embeddings",
    "artifacts/gatv2_v1/predictions",
]

RETURN_PACKAGE_TEMPLATE = [
    "artifacts/h200_mappo_training/<run_id>/environment_report.json",
    "artifacts/h200_mappo_training/<run_id>/training_config.json",
    "artifacts/h200_mappo_training/<run_id>/training_command.txt",
    "artifacts/h200_mappo_training/<run_id>/git_commit.txt",
    "artifacts/h200_mappo_training/<run_id>/logs/train_stdout.log",
    "artifacts/h200_mappo_training/<run_id>/logs/train_stderr.log",
    "artifacts/h200_mappo_training/<run_id>/logs/train_metrics.jsonl",
    "artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt",
    "artifacts/h200_mappo_training/<run_id>/checkpoints/checkpoint_manifest.json",
    "artifacts/h200_mappo_training/<run_id>/validation/validate_mappo_checkpoint_actual_report.json",
    "artifacts/h200_mappo_training/<run_id>/validation/h200_actual_checkpoint_preflight_report.json",
    "artifacts/h200_mappo_training/<run_id>/validation/h200_actual_checkpoint_folder_contract_report.json",
    "artifacts/h200_mappo_training/<run_id>/README_run_summary.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_git(project_root: Path, args: List[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=20,
        )
        return {
            "command": ["git", *args],
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "available": True,
        }
    except Exception as exc:
        return {
            "command": ["git", *args],
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "available": False,
        }


def git_summary(project_root: Path) -> Dict[str, Any]:
    head = run_git(project_root, ["rev-parse", "HEAD"])
    short = run_git(project_root, ["rev-parse", "--short", "HEAD"])
    branch = run_git(project_root, ["branch", "--show-current"])
    status = run_git(project_root, ["status", "--short"])
    remote = run_git(project_root, ["remote", "-v"])

    status_short = status.get("stdout", "")
    return {
        "git_available": bool(head.get("available")) and head.get("returncode") == 0,
        "pinned_commit": head.get("stdout", "").strip() if head.get("returncode") == 0 else None,
        "pinned_commit_short": short.get("stdout", "").strip() if short.get("returncode") == 0 else None,
        "branch": branch.get("stdout", "").strip() if branch.get("returncode") == 0 else None,
        "status_short": status_short,
        "working_tree_clean": status_short.strip() == "",
        "remote_v": remote.get("stdout", ""),
    }


def inspect_paths(project_root: Path, paths: List[str]) -> Dict[str, Any]:
    rows = []
    missing = []

    for rel in paths:
        path = project_root / rel
        exists = path.exists()
        is_file = path.is_file() if exists else False
        is_dir = path.is_dir() if exists else False

        rows.append({
            "path": rel,
            "exists": exists,
            "is_file": is_file,
            "is_dir": is_dir,
        })

        if not exists:
            missing.append(rel)

    return {
        "count": len(paths),
        "missing_count": len(missing),
        "missing": missing,
        "items": rows,
        "all_present": len(missing) == 0,
    }


def build_h200_command_pack(pinned_commit: str | None) -> List[Dict[str, Any]]:
    commit = pinned_commit or "<PINNED_COMMIT_HASH>"

    return [
        {
            "label": "checkout pinned commit",
            "command": f"git checkout {commit}",
        },
        {
            "label": "verify clean checkout",
            "command": "git status --short",
        },
        {
            "label": "create environment report",
            "command": (
                "python 05_training/policies/generate_h200_environment_report.py "
                "--project-root . "
                "--output-path artifacts/h200_mappo_training/<run_id>/environment_report.json "
                "--require-cuda --require-h200-name --require-clean-git"
            ),
        },
        {
            "label": "execute local validation bundle on H200",
            "command": (
                "python 05_training/policies/run_h200_training_local_validation_bundle.py "
                "--project-root . "
                "--report-path artifacts/h200_local_validation_bundle/step70_h200_execute_report.json "
                "--python-exe \"$(which python)\" "
                "--execute"
            ),
        },
        {
            "label": "run train_mappo_actual dry-run contract",
            "command": (
                "python 05_training/train_mappo_actual.py "
                "--condition-id A "
                "--experiment-contract artifacts/experiment_A_v1/experiment_A_contract.json "
                "--baseline-contract artifacts/baseline_v1/baseline_contract.json "
                "--output-root artifacts/h200_mappo_training/<run_id> "
                "--seeds 1 2 3 "
                "--device cuda "
                "--qwen-train false "
                "--qwen-inference false "
                "--qwen-trigger-rate 0.0 "
                "--reward-version mappo_reward_v1 "
                "--energy-proxy-model-version daegu_energy_proxy_v1 "
                "--k-dist-kwh-per-m 0.0012 "
                "--k-acc-kwh-per-event 0.1800 "
                "--k-idle-kwh-per-sec 0.0080 "
                "--dry-run-contract"
            ),
        },
        {
            "label": "validate actual checkpoint after training",
            "command": (
                "python 05_training/policies/validate_mappo_checkpoint.py "
                "--checkpoint artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt "
                "--mode actual"
            ),
        },
        {
            "label": "run Step 58 actual preflight after training",
            "command": (
                "python 05_training/policies/h200_actual_checkpoint_preflight_checklist.py "
                "--checkpoint-path artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt "
                "--mode actual "
                "--report-path artifacts/h200_mappo_training/<run_id>/validation/h200_actual_checkpoint_preflight_report.json"
            ),
        },
        {
            "label": "validate H200 checkpoint folder",
            "command": (
                "python 05_training/policies/validate_h200_actual_checkpoint_folder.py "
                "--run-root artifacts/h200_mappo_training/<run_id>"
            ),
        },
    ]


def build_manifest(project_root: Path, strict_artifacts: bool) -> Dict[str, Any]:
    git = git_summary(project_root)
    source_check = inspect_paths(project_root, REQUIRED_SOURCE_FILES)
    artifact_check = inspect_paths(project_root, REQUIRED_ARTIFACTS)
    gatv2_check = inspect_paths(project_root, CONDITIONAL_GATV2_ARTIFACTS)

    blockers: List[str] = []
    warnings: List[str] = []

    if not source_check["all_present"]:
        blockers.append(f"missing required source files: {source_check['missing']}")

    if not git.get("working_tree_clean", False):
        warnings.append("working tree is not clean; transfer should use committed pinned state only")

    if not artifact_check["all_present"]:
        msg = f"missing required artifacts: {artifact_check['missing']}"
        if strict_artifacts:
            blockers.append(msg)
        else:
            warnings.append(msg)

    if not gatv2_check["all_present"]:
        warnings.append(
            "conditional GATv2 artifacts are incomplete; actual MAPPO training must document the substitute or blocker"
        )

    if blockers:
        status = "BLOCKED"
    elif strict_artifacts:
        status = "TRANSFER_READY"
    else:
        status = "PLANNING_MANIFEST_READY"

    return {
        "artifact_version": "h200_transfer_package_manifest_v1_step74",
        "created_at_utc": utc_now(),
        "transfer_manifest_status": status,
        "strict_artifacts": bool(strict_artifacts),
        "project_root": str(project_root),
        "git": git,
        "source_file_check": source_check,
        "required_artifact_check": artifact_check,
        "conditional_gatv2_artifact_check": gatv2_check,
        "h200_command_pack": build_h200_command_pack(git.get("pinned_commit")),
        "return_package_template": RETURN_PACKAGE_TEMPLATE,
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": {
            "h200_transfer_manifest_ready": status in {"PLANNING_MANIFEST_READY", "TRANSFER_READY"},
            "actual_h200_training_completed": False,
            "trained_h200_checkpoint_exists": False,
            "causal_performance_claim_allowed": False,
            "note": "Step 74 creates a transfer package manifest only. It does not train MAPPO.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Step 74 H200 transfer package manifest."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Block if required transfer artifacts are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output_path)

    try:
        if not project_root.exists():
            raise TransferManifestError(f"project root not found: {project_root}")

        manifest = build_manifest(project_root, strict_artifacts=bool(args.strict_artifacts))
        dump_json(output_path, manifest)

        if manifest["transfer_manifest_status"] == "BLOCKED":
            raise TransferManifestError("; ".join(manifest["blockers"]))

        print("[OK] Step 74 H200 transfer package manifest generated")
        print(f"[OK] status        : {manifest['transfer_manifest_status']}")
        print(f"[OK] strict        : {manifest['strict_artifacts']}")
        print(f"[OK] output_path   : {output_path}")
        print(f"[OK] source_files  : {manifest['source_file_check']['count']}")
        print(f"[OK] artifacts     : {manifest['required_artifact_check']['count']}")
        print(f"[OK] warnings      : {len(manifest['warnings'])}")
        return 0

    except TransferManifestError as exc:
        failure = {
            "artifact_version": "h200_transfer_package_manifest_v1_step74",
            "created_at_utc": utc_now(),
            "transfer_manifest_status": "BLOCKED",
            "project_root": str(project_root),
            "error": str(exc),
        }
        try:
            dump_json(output_path, failure)
        except Exception:
            pass

        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] output_path: {output_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
