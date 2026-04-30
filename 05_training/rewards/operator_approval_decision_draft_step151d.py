from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ACCEPTED_STEP151C_STATUSES = {
    "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED",
    "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED",
}
EXPECTED_STEP151C_DECISION = "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED"

STEP151D_STATUS = "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED"
STEP151D_DECISION = "APPROVAL_DRAFT_ONLY_NOT_APPROVED_NOT_RELEASED"

DEFAULT_STEP151C_MANIFEST = (
    "artifacts/rewards/h200_actual_preflight_rerun_result_intake_placeholder_step151c/"
    "h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
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


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    checks = manifest.get("checks", [])
    lines = [
        "# Step 151-D operator approval decision draft summary",
        "",
        f"- approval_decision_status: {manifest.get('approval_decision_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
        f"- step151c_intake_status: {manifest.get('step151c_intake_status')}",
        f"- h200_expected_preflight_passed: {manifest.get('h200_expected_preflight_passed')}",
        f"- operator_approval_recorded: {manifest.get('operator_approval_recorded')}",
        f"- operator_approval_granted: {manifest.get('operator_approval_granted')}",
        f"- actual_execution_allowed: {manifest.get('actual_execution_allowed')}",
        f"- actual_execution_released: {manifest.get('actual_execution_released')}",
        f"- train_allowed: {manifest.get('train_allowed')}",
        f"- next_gate: {manifest.get('next_gate')}",
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
            "Step 151-D is an operator approval decision draft only.",
            "It does not record operator approval and does not release actual reward ablation execution.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_draft_manifest(
    project_root: Path,
    step151c_manifest_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    if not step151c_manifest_path.exists():
        raise FileNotFoundError(f"Step 151-C manifest not found: {step151c_manifest_path}")

    step151c = load_json(step151c_manifest_path)
    checks: List[Dict[str, Any]] = []

    step151c_status = str(step151c.get("intake_status", ""))
    step151c_decision = str(step151c.get("decision", ""))
    step151c_hard_failures = coerce_int(step151c.get("hard_failures", 999), default=999)

    step151c_actual_allowed = coerce_bool(step151c.get("actual_execution_allowed"), default=None)
    step151c_actual_released = coerce_bool(step151c.get("actual_execution_released"), default=None)
    step151c_train_allowed = coerce_bool(step151c.get("train_allowed"), default=None)
    step151c_operator_approval = coerce_bool(step151c.get("operator_approval_recorded"), default=None)

    h200_manifest_present = coerce_bool(step151c.get("h200_expected_preflight_manifest_present"), default=False)
    h200_preflight_passed = coerce_bool(step151c.get("h200_expected_preflight_passed"), default=False)

    add_check(
        checks,
        "step151c_status_accepted",
        step151c_status in ACCEPTED_STEP151C_STATUSES,
        "hard",
        f"Step 151-C intake_status must be one of {sorted(ACCEPTED_STEP151C_STATUSES)}; got {step151c_status}",
        {"step151c_manifest": str(step151c_manifest_path)},
    )

    add_check(
        checks,
        "step151c_decision_intake_only",
        step151c_decision == EXPECTED_STEP151C_DECISION,
        "hard",
        f"Step 151-C decision must be {EXPECTED_STEP151C_DECISION}; got {step151c_decision}",
    )

    add_check(
        checks,
        "step151c_hard_failures_zero",
        step151c_hard_failures == 0,
        "hard",
        f"Step 151-C hard_failures must be 0; got {step151c_hard_failures}",
    )

    add_check(
        checks,
        "step151c_execution_still_locked",
        step151c_actual_allowed is False and step151c_actual_released is False and step151c_train_allowed is False,
        "hard",
        "Step 151-C must keep actual_execution_allowed=false, actual_execution_released=false, train_allowed=false",
        {
            "actual_execution_allowed": step151c_actual_allowed,
            "actual_execution_released": step151c_actual_released,
            "train_allowed": step151c_train_allowed,
        },
    )

    add_check(
        checks,
        "step151c_operator_approval_not_recorded",
        step151c_operator_approval is False,
        "hard",
        "Step 151-C must not record operator approval",
        {"operator_approval_recorded": step151c_operator_approval},
    )

    if step151c_status == "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED":
        add_check(
            checks,
            "h200_preflight_result_waiting_acceptance",
            h200_manifest_present is False and h200_preflight_passed is False,
            "hard",
            "Waiting Step 151-C state is accepted only when H200 result is not present and not passed",
            {
                "h200_expected_preflight_manifest_present": h200_manifest_present,
                "h200_expected_preflight_passed": h200_preflight_passed,
            },
        )

    if step151c_status == "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED":
        add_check(
            checks,
            "h200_preflight_result_validated_acceptance",
            h200_manifest_present is True and h200_preflight_passed is True,
            "hard",
            "Validated Step 151-C state is accepted only when H200 result is present and passed",
            {
                "h200_expected_preflight_manifest_present": h200_manifest_present,
                "h200_expected_preflight_passed": h200_preflight_passed,
            },
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
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }

    for key, value in lock_values.items():
        add_check(
            checks,
            f"lock_{key}_false",
            value is False,
            "hard",
            f"{key} must remain false in Step 151-D",
            {"value": value},
        )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    approval_decision_status = STEP151D_STATUS if hard_failures == 0 else "BLOCKED"

    if hard_failures != 0:
        next_gate = "BLOCKED_UNTIL_STEP151C_FIXED"
    elif h200_preflight_passed is True:
        next_gate = "READY_FOR_STEP151E_OPERATOR_APPROVAL_RECORD_DRAFT_STILL_LOCKED"
    else:
        next_gate = "WAITING_FOR_H200_PREFLIGHT_RESULT_BEFORE_APPROVAL_RECORD"

    manifest: Dict[str, Any] = {
        "artifact_version": "operator_approval_decision_draft_step151d_v1",
        "step": "151-D",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "input_step": "151-C",
        "step151c_manifest": str(step151c_manifest_path),
        "step151c_intake_status": step151c_status,
        "approval_decision_status": approval_decision_status,
        "decision": STEP151D_DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "h200_expected_preflight_manifest_present": bool(h200_manifest_present),
        "h200_expected_preflight_passed": bool(h200_preflight_passed),
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
        "operator_approval_decision_final": False,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "approval_scope": {
            "target_matrix": "A/A90/A80/A70 x R0-R5 x seeds 1-3",
            "target_run_count": 72,
            "baselines_required": ["B0R", "B1", "B2"],
            "approval_type": "draft_only_still_locked",
        },
        "required_before_operator_approval_record": [
            "h200_expected_preflight_passed_true",
            "real_h200_step149_manifest_archived",
            "operator_identity_recorded",
            "approval_timestamp_recorded",
            "approved_git_commit_recorded",
            "output_root_empty_or_archived",
            "separate_actual_execution_release_manifest",
        ],
        "actual_release_requires_separate_manifest": True,
        "next_gate": next_gate,
        "notes": [
            "Step 151-D is an operator approval decision draft only.",
            "It intentionally does not record operator approval.",
            "It intentionally does not release actual reward ablation execution.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "operator_approval_decision_draft_step151d_manifest.json"
    summary_path = output_root / "operator_approval_decision_draft_step151d_summary.md"

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
    parser.add_argument("--step151c-manifest", default=DEFAULT_STEP151C_MANIFEST)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/operator_approval_decision_draft_step151d",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    step151c_manifest_path = Path(args.step151c_manifest)
    if not step151c_manifest_path.is_absolute():
        step151c_manifest_path = project_root / step151c_manifest_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    manifest = build_draft_manifest(
        project_root=project_root,
        step151c_manifest_path=step151c_manifest_path,
        output_root=output_root,
    )

    print("[OK] Step 151-D operator approval decision draft completed")
    print(f"[OK] approval_decision_status : {manifest['approval_decision_status']}")
    print(f"[OK] decision                 : {manifest['decision']}")
    print(f"[OK] hard_failures            : {manifest['hard_failures']}")
    print(f"[OK] warnings                 : {manifest['warnings']}")
    print(f"[OK] step151c_intake_status   : {manifest['step151c_intake_status']}")
    print(f"[OK] h200_expected_preflight_passed : {manifest['h200_expected_preflight_passed']}")
    print(f"[OK] operator_approval_recorded     : {manifest['operator_approval_recorded']}")
    print(f"[OK] actual_execution_allowed       : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released      : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed                  : {manifest['train_allowed']}")
    print(f"[OK] manifest                 : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
