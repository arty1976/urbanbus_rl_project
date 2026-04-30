from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_STEP151B_STATUS = "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED"
EXPECTED_STEP151B_DECISION = "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED"

STATUS_WAITING = "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED"
STATUS_RECEIVED_VALIDATED = "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED"
STATUS_BLOCKED = "BLOCKED"

DECISION = "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED"

DEFAULT_STEP151B_MANIFEST = (
    "artifacts/rewards/h200_expected_preflight_rerun_checklist_step151b/"
    "h200_expected_preflight_rerun_checklist_step151b_manifest.json"
)

DEFAULT_H200_STEP149_MANIFEST = (
    "artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual/"
    "h200_environment_preflight_result_manifest_step149.json"
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
        stripped = value.strip()
        if stripped == "":
            return default
        try:
            return int(float(stripped))
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


def analyze_h200_step149_manifest(path: Path, min_gpu_count: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "audit_status": None,
            "environment_status": None,
            "expect_h200": None,
            "torch_import_ok": None,
            "cuda_available": None,
            "gpu_count": None,
            "hard_failures": None,
            "actual_execution_allowed": None,
            "train_allowed": None,
            "passed": False,
            "waiting": True,
        }

    payload = load_json(path)

    audit_status = str(payload.get("audit_status", ""))
    environment_status = str(payload.get("environment_status", ""))
    expect_h200 = coerce_bool(payload.get("expect_h200"), default=None)
    torch_import_ok = coerce_bool(payload.get("torch_import_ok"), default=None)
    cuda_available = coerce_bool(payload.get("cuda_available"), default=None)
    gpu_count = coerce_int(payload.get("gpu_count"), default=-1)
    hard_failures = coerce_int(payload.get("hard_failures"), default=999)
    actual_execution_allowed = coerce_bool(payload.get("actual_execution_allowed"), default=None)
    train_allowed = coerce_bool(payload.get("train_allowed"), default=None)

    passed = (
        audit_status == "PASS"
        and expect_h200 is True
        and torch_import_ok is True
        and cuda_available is True
        and gpu_count >= int(min_gpu_count)
        and hard_failures == 0
        and actual_execution_allowed is False
        and train_allowed is False
    )

    return {
        "path": str(path),
        "exists": True,
        "audit_status": audit_status,
        "environment_status": environment_status,
        "expect_h200": expect_h200,
        "torch_import_ok": torch_import_ok,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "min_gpu_count": int(min_gpu_count),
        "hard_failures": hard_failures,
        "actual_execution_allowed": actual_execution_allowed,
        "train_allowed": train_allowed,
        "passed": passed,
        "waiting": False,
    }


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    checks = manifest.get("checks", [])
    lines = [
        "# Step 151-C H200 actual preflight rerun result intake placeholder summary",
        "",
        f"- intake_status: {manifest.get('intake_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
        f"- h200_expected_preflight_manifest_present: {manifest.get('h200_expected_preflight_manifest_present')}",
        f"- h200_expected_preflight_passed: {manifest.get('h200_expected_preflight_passed')}",
        f"- actual_execution_allowed: {manifest.get('actual_execution_allowed')}",
        f"- actual_execution_released: {manifest.get('actual_execution_released')}",
        f"- train_allowed: {manifest.get('train_allowed')}",
        f"- next_gate: {manifest.get('next_gate')}",
        "",
        "## Expected H200 result manifest",
        "",
        f"```text\n{manifest.get('h200_step149_expected_manifest')}\n```",
        "",
        "## Required H200 command",
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
            "Step 151-C is an intake placeholder only. It does not release actual reward ablation execution.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_intake_manifest(
    project_root: Path,
    step151b_manifest_path: Path,
    h200_step149_manifest_path: Path,
    output_root: Path,
    min_gpu_count: int,
) -> Dict[str, Any]:
    if not step151b_manifest_path.exists():
        raise FileNotFoundError(f"Step 151-B manifest not found: {step151b_manifest_path}")

    step151b = load_json(step151b_manifest_path)
    checks: List[Dict[str, Any]] = []

    step151b_status = str(step151b.get("checklist_status", ""))
    step151b_decision = str(step151b.get("decision", ""))
    step151b_hard_failures = coerce_int(step151b.get("hard_failures", 999), default=999)
    step151b_actual_allowed = coerce_bool(step151b.get("actual_execution_allowed"), default=None)
    step151b_actual_released = coerce_bool(step151b.get("actual_execution_released"), default=None)
    step151b_train_allowed = coerce_bool(step151b.get("train_allowed"), default=None)
    step151b_rerun_executed = coerce_bool(step151b.get("h200_expected_preflight_rerun_executed"), default=None)
    step151b_h200_passed = coerce_bool(step151b.get("h200_expected_preflight_passed"), default=None)

    add_check(
        checks,
        "step151b_status_ready_still_locked",
        step151b_status == EXPECTED_STEP151B_STATUS,
        "hard",
        f"Step 151-B checklist_status must be {EXPECTED_STEP151B_STATUS}; got {step151b_status}",
        {"step151b_manifest": str(step151b_manifest_path)},
    )

    add_check(
        checks,
        "step151b_decision_checklist_only",
        step151b_decision == EXPECTED_STEP151B_DECISION,
        "hard",
        f"Step 151-B decision must be {EXPECTED_STEP151B_DECISION}; got {step151b_decision}",
    )

    add_check(
        checks,
        "step151b_hard_failures_zero",
        step151b_hard_failures == 0,
        "hard",
        f"Step 151-B hard_failures must be 0; got {step151b_hard_failures}",
    )

    add_check(
        checks,
        "step151b_execution_still_locked",
        step151b_actual_allowed is False and step151b_actual_released is False and step151b_train_allowed is False,
        "hard",
        "Step 151-B must keep actual_execution_allowed=false, actual_execution_released=false, train_allowed=false",
        {
            "actual_execution_allowed": step151b_actual_allowed,
            "actual_execution_released": step151b_actual_released,
            "train_allowed": step151b_train_allowed,
        },
    )

    add_check(
        checks,
        "step151b_did_not_claim_h200_pass",
        step151b_rerun_executed is False and step151b_h200_passed is False,
        "hard",
        "Step 151-B must not claim that H200 expected preflight already executed or passed",
        {
            "h200_expected_preflight_rerun_executed": step151b_rerun_executed,
            "h200_expected_preflight_passed": step151b_h200_passed,
        },
    )

    h200 = analyze_h200_step149_manifest(h200_step149_manifest_path, min_gpu_count=min_gpu_count)

    if not h200["exists"]:
        add_check(
            checks,
            "h200_step149_result_manifest_waiting",
            True,
            "hard",
            "H200 Step 149 expected result manifest is not present yet; intake placeholder is waiting",
            h200,
        )
    else:
        add_check(
            checks,
            "h200_step149_result_manifest_received_and_valid",
            h200["passed"],
            "hard",
            "Received H200 Step 149 result must PASS with expect_h200=true, cuda_available=true, gpu_count>=min_gpu_count, and execution still locked",
            h200,
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
        "operator_approval_recorded": False,
    }

    for key, value in lock_values.items():
        add_check(
            checks,
            f"lock_{key}_false",
            value is False,
            "hard",
            f"{key} must remain false in Step 151-C",
            {"value": value},
        )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    if hard_failures != 0:
        intake_status = STATUS_BLOCKED
    elif h200["exists"] and h200["passed"]:
        intake_status = STATUS_RECEIVED_VALIDATED
    else:
        intake_status = STATUS_WAITING

    if intake_status == STATUS_RECEIVED_VALIDATED:
        next_gate = "READY_FOR_STEP151D_OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED"
    elif intake_status == STATUS_WAITING:
        next_gate = "READY_FOR_REAL_H200_STEP149_RESULT_UPLOAD_OR_SYNC"
    else:
        next_gate = "BLOCKED_UNTIL_H200_STEP149_RESULT_FIXED"

    manifest: Dict[str, Any] = {
        "artifact_version": "h200_actual_preflight_rerun_result_intake_placeholder_step151c_v1",
        "step": "151-C",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "input_step": "151-B",
        "step151b_manifest": str(step151b_manifest_path),
        "h200_step149_expected_manifest": str(h200_step149_manifest_path),
        "h200_step149_expected_min_gpu_count": int(min_gpu_count),
        "intake_status": intake_status,
        "decision": DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "h200_expected_preflight_manifest_present": bool(h200["exists"]),
        "h200_expected_preflight_rerun_executed": bool(h200["exists"]),
        "h200_expected_preflight_passed": bool(h200["exists"] and h200["passed"]),
        "h200_result_summary": h200,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "operator_approval_recorded": False,
        "h200_preflight_result_intake_placeholder_ready": True,
        "actual_release_requires_separate_manifest": True,
        "required_before_actual_release": [
            "h200_expected_preflight_manifest_present_true",
            "h200_expected_preflight_passed_true",
            "operator_approval_recorded_true",
            "actual_execution_release_manifest_separate_from_intake_placeholder",
            "output_root_empty_or_archived",
            "git_commit_pinned",
        ],
        "next_gate": next_gate,
        "notes": [
            "Step 151-C is an intake placeholder and optional validator for the future H200 Step 149 rerun result.",
            "If the H200 result manifest is not present, WAITING status is valid.",
            "If the H200 result manifest is present but invalid, the step is BLOCKED.",
            "Actual reward ablation execution remains locked.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
    summary_path = output_root / "h200_actual_preflight_rerun_result_intake_placeholder_step151c_summary.md"

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
    parser.add_argument("--step151b-manifest", default=DEFAULT_STEP151B_MANIFEST)
    parser.add_argument("--h200-step149-manifest", default=DEFAULT_H200_STEP149_MANIFEST)
    parser.add_argument("--min-gpu-count", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_actual_preflight_rerun_result_intake_placeholder_step151c",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    step151b_manifest_path = Path(args.step151b_manifest)
    if not step151b_manifest_path.is_absolute():
        step151b_manifest_path = project_root / step151b_manifest_path

    h200_step149_manifest_path = Path(args.h200_step149_manifest)
    if not h200_step149_manifest_path.is_absolute():
        h200_step149_manifest_path = project_root / h200_step149_manifest_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    manifest = build_intake_manifest(
        project_root=project_root,
        step151b_manifest_path=step151b_manifest_path,
        h200_step149_manifest_path=h200_step149_manifest_path,
        output_root=output_root,
        min_gpu_count=args.min_gpu_count,
    )

    print("[OK] Step 151-C H200 actual preflight rerun result intake placeholder completed")
    print(f"[OK] intake_status  : {manifest['intake_status']}")
    print(f"[OK] decision       : {manifest['decision']}")
    print(f"[OK] hard_failures  : {manifest['hard_failures']}")
    print(f"[OK] warnings       : {manifest['warnings']}")
    print(f"[OK] h200_expected_preflight_manifest_present : {manifest['h200_expected_preflight_manifest_present']}")
    print(f"[OK] h200_expected_preflight_passed          : {manifest['h200_expected_preflight_passed']}")
    print(f"[OK] actual_execution_allowed  : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed             : {manifest['train_allowed']}")
    print(f"[OK] manifest      : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
