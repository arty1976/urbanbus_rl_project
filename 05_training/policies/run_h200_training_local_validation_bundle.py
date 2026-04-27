from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class BundleError(RuntimeError):
    pass


REQUIRED_FILES = [
    "05_training/policies/h200_actual_checkpoint_preflight_checklist.py",
    "05_training/policies/test_h200_actual_checkpoint_preflight_checklist.py",
    "05_training/policies/a_family_mappo_actual_execution_plan.py",
    "05_training/policies/test_a_family_mappo_actual_execution_plan.py",
    "05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py",
    "05_training/policies/test_run_a_family_mappo_actual_matrix_from_plan.py",
    "05_training/policies/H200_actual_mappo_final_runbook.md",
    "05_training/policies/test_h200_actual_mappo_final_runbook.py",
    "05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py",
    "05_training/policies/test_h200_actual_checkpoint_arrival_workflow.py",
    "05_training/policies/MAPPO_actual_reproducibility_note.md",
    "05_training/policies/test_mappo_actual_reproducibility_note.py",
    "05_training/policies/H200_mappo_training_runbook.md",
    "05_training/policies/test_h200_mappo_training_runbook.py",
    "05_training/policies/generate_h200_environment_report.py",
    "05_training/policies/test_generate_h200_environment_report.py",
    "05_training/policies/validate_h200_actual_checkpoint_folder.py",
    "05_training/policies/test_validate_h200_actual_checkpoint_folder.py",
    "05_training/train_mappo_actual.py",
    "05_training/policies/test_train_mappo_actual_contract.py",
]

SELF_TESTS = [
    "05_training/policies/test_h200_actual_checkpoint_preflight_checklist.py",
    "05_training/policies/test_a_family_mappo_actual_execution_plan.py",
    "05_training/policies/test_run_a_family_mappo_actual_matrix_from_plan.py",
    "05_training/policies/test_h200_actual_mappo_final_runbook.py",
    "05_training/policies/test_h200_actual_checkpoint_arrival_workflow.py",
    "05_training/policies/test_mappo_actual_reproducibility_note.py",
    "05_training/policies/test_h200_mappo_training_runbook.py",
    "05_training/policies/test_generate_h200_environment_report.py",
    "05_training/policies/test_validate_h200_actual_checkpoint_folder.py",
    "05_training/policies/test_train_mappo_actual_contract.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_command(command: List[str], cwd: Path) -> Dict[str, Any]:
    started_at = utc_now()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    ended_at = utc_now()

    return {
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def check_required_files(project_root: Path) -> Dict[str, Any]:
    rows = []
    missing = []

    for rel in REQUIRED_FILES:
        path = project_root / rel
        exists = path.exists()
        rows.append({
            "path": rel,
            "exists": exists,
            "is_file": path.is_file() if exists else False,
        })
        if not exists or not path.is_file():
            missing.append(rel)

    return {
        "required_count": len(REQUIRED_FILES),
        "missing_count": len(missing),
        "missing_files": missing,
        "files": rows,
        "passed": len(missing) == 0,
    }


def build_command_pack(python_exe: str) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []

    compile_targets = [
        "05_training/policies/h200_actual_checkpoint_preflight_checklist.py",
        "05_training/policies/a_family_mappo_actual_execution_plan.py",
        "05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py",
        "05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py",
        "05_training/policies/generate_h200_environment_report.py",
        "05_training/policies/validate_h200_actual_checkpoint_folder.py",
        "05_training/train_mappo_actual.py",
    ]

    commands.append({
        "label": "py_compile core H200 MAPPO gate scripts",
        "command": [python_exe, "-m", "py_compile", *compile_targets],
        "kind": "compile",
    })

    for test in SELF_TESTS:
        commands.append({
            "label": f"self-test {test}",
            "command": [python_exe, test],
            "kind": "self_test",
        })

    return commands


def execute_command_pack(
    command_pack: List[Dict[str, Any]],
    project_root: Path,
) -> List[Dict[str, Any]]:
    logs = []

    for item in command_pack:
        log = run_command(item["command"], cwd=project_root)
        logs.append({
            "label": item["label"],
            "kind": item["kind"],
            **log,
        })

        if log["returncode"] != 0:
            raise BundleError(
                f"command failed: {item['label']} returncode={log['returncode']}"
            )

    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 70 H200 training local validation bundle."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run == args.execute:
        print("[FAIL] Choose exactly one mode: --dry-run or --execute", file=sys.stderr)
        return 2

    project_root = Path(args.project_root).resolve()
    report_path = Path(args.report_path)

    try:
        if not project_root.exists():
            raise BundleError(f"project root not found: {project_root}")

        required_file_check = check_required_files(project_root)
        if not required_file_check["passed"]:
            raise BundleError(f"missing required files: {required_file_check['missing_files']}")

        command_pack = build_command_pack(args.python_exe)

        execution_log: List[Dict[str, Any]] = []
        if args.execute:
            execution_log = execute_command_pack(command_pack, project_root)

        report = {
            "artifact_version": "h200_training_local_validation_bundle_v1_step70",
            "created_at_utc": utc_now(),
            "bundle_status": "DRY_RUN_VALIDATED" if args.dry_run else "EXECUTION_VALIDATED",
            "project_root": str(project_root),
            "python_exe": args.python_exe,
            "dry_run": bool(args.dry_run),
            "executed": bool(args.execute),
            "required_file_check": required_file_check,
            "command_count": len(command_pack),
            "command_pack": command_pack,
            "execution_log": execution_log,
            "claim_boundary": {
                "h200_training_scripts_ready": True,
                "trained_h200_checkpoint_exists": False,
                "causal_performance_claim_allowed": False,
                "note": "Step 70 validates local gate scripts only. It does not train MAPPO.",
            },
        }

        dump_json(report_path, report)

        print("[OK] Step 70 H200 training local validation bundle PASS")
        print(f"[OK] mode          : {'dry-run' if args.dry_run else 'execute'}")
        print(f"[OK] report_path   : {report_path}")
        print(f"[OK] command_count : {len(command_pack)}")
        print(f"[OK] status        : {report['bundle_status']}")
        return 0

    except BundleError as exc:
        failure = {
            "artifact_version": "h200_training_local_validation_bundle_v1_step70",
            "created_at_utc": utc_now(),
            "bundle_status": "BLOCKED",
            "project_root": str(project_root),
            "error": str(exc),
        }
        try:
            dump_json(report_path, failure)
        except Exception:
            pass

        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] report_path: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
