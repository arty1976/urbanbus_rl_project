from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_STEP151A_STATUS = "RELEASE_MANIFEST_DRAFT_STILL_LOCKED"
EXPECTED_STEP151A_DECISION = "DRAFT_ONLY_NOT_RELEASED"
EXPECTED_H200_ARGS = "--expect-h200 --min-gpu-count 1"

STEP151B_STATUS = "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED"
STEP151B_DECISION = "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED"
NEXT_GATE = "READY_FOR_REAL_H200_STEP149_EXPECT_H200_RERUN"

DEFAULT_STEP151A_MANIFEST = (
    "artifacts/rewards/explicit_operator_release_manifest_draft_step151a/"
    "explicit_operator_release_manifest_draft_step151a_manifest.json"
)

DEFAULT_STEP149_LOCAL_MANIFEST = (
    "artifacts/rewards/h200_environment_preflight_result_manifest_step149/"
    "h200_environment_preflight_result_manifest_step149.json"
)

REQUIRED_H200_COMMAND = (
    "python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py "
    "--project-root /workspace/urbanbus_rl_project "
    "--expect-h200 --min-gpu-count 1 "
    "--output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual"
)



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")



def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def coerce_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    return default



def coerce_int(value: Any, default: int = 999) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except Exception:
            return default
    if isinstance(value, list):
        return len(value)
    return default



def get_git_commit(project_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"



def get_git_status_short(project_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"



def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    passed: bool,
    severity: str,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "severity": severity,
            "message": message,
            "evidence": evidence or {},
        }
    )



def analyze_step149_local(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "audit_status": None,
            "environment_status": None,
            "expect_h200": None,
            "hard_failures": None,
            "actual_execution_allowed": None,
            "train_allowed": None,
            "local_record_mode": None,
        }

    payload = load_json(path)

    audit_status = payload.get("audit_status")
    environment_status = payload.get("environment_status")
    expect_h200 = coerce_bool(payload.get("expect_h200"), default=None)
    hard_failures = coerce_int(payload.get("hard_failures"), default=999)
    actual_execution_allowed = coerce_bool(payload.get("actual_execution_allowed"), default=None)
    train_allowed = coerce_bool(payload.get("train_allowed"), default=None)

    env_text = str(environment_status or "").upper()
    local_record_mode = (
        expect_h200 is False
        or "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED" in env_text
        or "LOCAL" in env_text
    )

    return {
        "path": str(path),
        "exists": True,
        "audit_status": audit_status,
        "environment_status": environment_status,
        "expect_h200": expect_h200,
        "hard_failures": hard_failures,
        "actual_execution_allowed": actual_execution_allowed,
        "train_allowed": train_allowed,
        "local_record_mode": local_record_mode,
    }



def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    checks = manifest.get("checks", [])
    lines = [
        "# Step 151-B H200 expected preflight rerun checklist summary",
        "",
        f"- checklist_status: {manifest.get('checklist_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
        f"- actual_execution_allowed: {manifest.get('actual_execution_allowed')}",
        f"- actual_execution_released: {manifest.get('actual_execution_released')}",
        f"- train_allowed: {manifest.get('train_allowed')}",
        f"- h200_expected_preflight_rerun_executed: {manifest.get('h200_expected_preflight_rerun_executed')}",
        f"- h200_expected_preflight_passed: {manifest.get('h200_expected_preflight_passed')}",
        f"- next_gate: {manifest.get('next_gate')}",
        "",
        "## Required H200 rerun command",
        "",
        "```bash",
        "python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \\",
        "  --project-root /workspace/urbanbus_rl_project \\",
        "  --expect-h200 \\",
        "  --min-gpu-count 1 \\",
        "  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual",
        "```",
        "",
        "## Checks",
        "",
    ]

    for check in checks:
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(
            f"- [{mark}] {check.get('check_id')} | {check.get('severity')} | {check.get('message')}"
        )

    lines.extend(
        [
            "",
            "## Lock statement",
            "",
            "Step 151-B is a checklist only. It does not run the H200 preflight and does not release actual reward ablation execution.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")



def build_checklist(
    project_root: Path,
    step151a_manifest_path: Path,
    step149_local_manifest_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    if not step151a_manifest_path.exists():
        raise FileNotFoundError(f"Step 151-A manifest not found: {step151a_manifest_path}")

    step151a = load_json(step151a_manifest_path)
    checks: List[Dict[str, Any]] = []

    step151a_status = str(step151a.get("release_manifest_status", ""))
    step151a_decision = str(step151a.get("decision", ""))
    step151a_hard_failures = coerce_int(step151a.get("hard_failures", 999), default=999)
    step151a_actual_allowed = coerce_bool(step151a.get("actual_execution_allowed"), default=None)
    step151a_actual_released = coerce_bool(step151a.get("actual_execution_released"), default=None)
    step151a_train_allowed = coerce_bool(step151a.get("train_allowed"), default=None)
    step151a_h200_required = coerce_bool(step151a.get("h200_expected_preflight_required"), default=None)
    step151a_h200_args = str(step151a.get("h200_expected_preflight_required_args", ""))

    add_check(
        checks,
        "step151a_status_still_locked",
        step151a_status == EXPECTED_STEP151A_STATUS,
        "hard",
        f"Step 151-A release_manifest_status must be {EXPECTED_STEP151A_STATUS}; got {step151a_status}",
        {"step151a_manifest": str(step151a_manifest_path)},
    )

    add_check(
        checks,
        "step151a_decision_draft_only",
        step151a_decision == EXPECTED_STEP151A_DECISION,
        "hard",
        f"Step 151-A decision must be {EXPECTED_STEP151A_DECISION}; got {step151a_decision}",
    )

    add_check(
        checks,
        "step151a_hard_failures_zero",
        step151a_hard_failures == 0,
        "hard",
        f"Step 151-A hard_failures must be 0; got {step151a_hard_failures}",
    )

    add_check(
        checks,
        "step151a_execution_still_locked",
        step151a_actual_allowed is False and step151a_actual_released is False and step151a_train_allowed is False,
        "hard",
        "Step 151-A must keep actual_execution_allowed=false, actual_execution_released=false, train_allowed=false",
        {
            "actual_execution_allowed": step151a_actual_allowed,
            "actual_execution_released": step151a_actual_released,
            "train_allowed": step151a_train_allowed,
        },
    )

    add_check(
        checks,
        "step151a_h200_expected_preflight_required",
        step151a_h200_required is True and step151a_h200_args == EXPECTED_H200_ARGS,
        "hard",
        f"Step 151-A must require H200 expected preflight rerun args: {EXPECTED_H200_ARGS}",
        {
            "h200_expected_preflight_required": step151a_h200_required,
            "h200_expected_preflight_required_args": step151a_h200_args,
        },
    )

    step149_local = analyze_step149_local(step149_local_manifest_path)

    add_check(
        checks,
        "step149_local_manifest_recorded",
        step149_local["exists"] is True,
        "warning",
        "Prior Step 149 local environment manifest should be recorded for traceability",
        step149_local,
    )

    if step149_local["exists"]:
        add_check(
            checks,
            "step149_local_record_mode_not_h200",
            step149_local["local_record_mode"] is True,
            "warning",
            "Existing Step 149 result appears to be local record mode; real H200 rerun is still required",
            step149_local,
        )

    add_check(
        checks,
        "required_h200_rerun_command_recorded",
        True,
        "hard",
        "H200 rerun command is recorded with --expect-h200 --min-gpu-count 1",
        {"command": REQUIRED_H200_COMMAND},
    )

    git_status_short = get_git_status_short(project_root)
    add_check(
        checks,
        "git_status_recorded",
        git_status_short != "UNKNOWN",
        "warning",
        "Git worktree status is recorded for operator review",
        {"git_status_short": git_status_short},
    )

    lock_values = {
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "h200_expected_preflight_rerun_executed": False,
        "h200_expected_preflight_passed": False,
    }

    for key, value in lock_values.items():
        add_check(
            checks,
            f"lock_{key}_false",
            value is False,
            "hard",
            f"{key} must remain false in Step 151-B checklist",
            {"value": value},
        )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    checklist_status = STEP151B_STATUS if hard_failures == 0 else "BLOCKED"

    manifest: Dict[str, Any] = {
        "artifact_version": "h200_expected_preflight_rerun_checklist_step151b_v1",
        "step": "151-B",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "input_step": "151-A",
        "step151a_manifest": str(step151a_manifest_path),
        "step149_local_manifest": str(step149_local_manifest_path),
        "step151a_release_manifest_status": step151a_status,
        "checklist_status": checklist_status,
        "decision": STEP151B_DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "h200_rerun_command": REQUIRED_H200_COMMAND,
        "h200_rerun_expected_project_root": "/workspace/urbanbus_rl_project",
        "h200_rerun_expected_output_root": "artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual",
        "h200_expected_preflight_required": True,
        "h200_expected_preflight_required_args": EXPECTED_H200_ARGS,
        "h200_expected_preflight_rerun_executed": False,
        "h200_expected_preflight_passed": False,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "required_h200_step149_pass_conditions": {
            "audit_status": "PASS",
            "expect_h200": True,
            "cuda_available": True,
            "gpu_count_minimum": 1,
            "hard_failures": 0,
            "actual_execution_allowed": False,
            "train_allowed": False,
        },
        "next_gate": NEXT_GATE,
        "notes": [
            "Step 151-B is a checklist only.",
            "It records the required real H200 Step 149 rerun command.",
            "It intentionally keeps actual execution locked.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "h200_expected_preflight_rerun_checklist_step151b_manifest.json"
    summary_path = output_root / "h200_expected_preflight_rerun_checklist_step151b_summary.md"

    manifest["output_files"] = {
        "manifest": str(manifest_path),
        "summary": str(summary_path),
    }

    dump_json(manifest_path, manifest)
    write_summary(summary_path, manifest)

    return manifest



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--step151a-manifest", default=DEFAULT_STEP151A_MANIFEST)
    parser.add_argument("--step149-local-manifest", default=DEFAULT_STEP149_LOCAL_MANIFEST)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_expected_preflight_rerun_checklist_step151b",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    step151a_manifest_path = Path(args.step151a_manifest)
    if not step151a_manifest_path.is_absolute():
        step151a_manifest_path = project_root / step151a_manifest_path

    step149_local_manifest_path = Path(args.step149_local_manifest)
    if not step149_local_manifest_path.is_absolute():
        step149_local_manifest_path = project_root / step149_local_manifest_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    manifest = build_checklist(
        project_root=project_root,
        step151a_manifest_path=step151a_manifest_path,
        step149_local_manifest_path=step149_local_manifest_path,
        output_root=output_root,
    )

    print("[OK] Step 151-B H200 expected preflight rerun checklist completed")
    print(f"[OK] checklist_status : {manifest['checklist_status']}")
    print(f"[OK] decision         : {manifest['decision']}")
    print(f"[OK] hard_failures    : {manifest['hard_failures']}")
    print(f"[OK] warnings         : {manifest['warnings']}")
    print(f"[OK] actual_execution_allowed  : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed             : {manifest['train_allowed']}")
    print(f"[OK] h200_expected_preflight_rerun_executed : {manifest['h200_expected_preflight_rerun_executed']}")
    print(f"[OK] h200_expected_preflight_passed         : {manifest['h200_expected_preflight_passed']}")
    print(f"[OK] manifest        : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)



if __name__ == "__main__":
    main()
