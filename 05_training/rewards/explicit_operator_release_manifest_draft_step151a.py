from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_STEP150_STATUS = "READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT"
EXPECTED_STEP150_DECISION = "CHECKLIST_ONLY_NOT_RELEASED"

STEP151A_STATUS = "RELEASE_MANIFEST_DRAFT_STILL_LOCKED"
STEP151A_DECISION = "DRAFT_ONLY_NOT_RELEASED"
NEXT_GATE = "READY_FOR_STEP151B_OPERATOR_APPROVAL_DRAFT_OR_H200_EXPECTED_PREFLIGHT_RERUN"

DEFAULT_STEP150_MANIFEST = (
    "artifacts/rewards/actual_reward_ablation_operator_release_checklist_step150/"
    "actual_reward_ablation_operator_release_checklist_step150_manifest.json"
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


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    checks = manifest.get("checks", [])
    lines = [
        "# Step 151-A explicit operator release manifest draft summary",
        "",
        f"- release_manifest_status: {manifest.get('release_manifest_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
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
            "Step 151-A is a draft-only release manifest. It does not release actual execution.",
            "",
            "Before any actual H200 run, Step 149 must be rerun on the real H200 server with:",
            "",
            "```text",
            "--expect-h200 --min-gpu-count 1",
            "```",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(
    project_root: Path,
    step150_manifest_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    if not step150_manifest_path.exists():
        raise FileNotFoundError(f"Step 150 manifest not found: {step150_manifest_path}")

    step150 = load_json(step150_manifest_path)
    checks: List[Dict[str, Any]] = []

    step150_status = str(step150.get("checklist_status", ""))
    step150_decision = str(step150.get("decision", ""))
    step150_hard_failures = coerce_int(step150.get("hard_failures", 999), default=999)
    step150_actual_allowed = coerce_bool(step150.get("actual_execution_allowed"), default=None)
    step150_train_allowed = coerce_bool(step150.get("train_allowed"), default=None)

    add_check(
        checks,
        "step150_status_ready_for_step151_draft",
        step150_status == EXPECTED_STEP150_STATUS,
        "hard",
        f"Step 150 checklist_status must be {EXPECTED_STEP150_STATUS}; got {step150_status}",
        {"step150_manifest": str(step150_manifest_path)},
    )

    add_check(
        checks,
        "step150_decision_checklist_only",
        step150_decision == EXPECTED_STEP150_DECISION,
        "hard",
        f"Step 150 decision must be {EXPECTED_STEP150_DECISION}; got {step150_decision}",
    )

    add_check(
        checks,
        "step150_hard_failures_zero",
        step150_hard_failures == 0,
        "hard",
        f"Step 150 hard_failures must be 0; got {step150_hard_failures}",
    )

    add_check(
        checks,
        "step150_actual_execution_locked",
        step150_actual_allowed is False,
        "hard",
        f"Step 150 actual_execution_allowed must be false; got {step150_actual_allowed}",
    )

    add_check(
        checks,
        "step150_train_locked",
        step150_train_allowed is False,
        "hard",
        f"Step 150 train_allowed must be false; got {step150_train_allowed}",
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

    add_check(
        checks,
        "h200_expected_preflight_rerun_required",
        True,
        "hard",
        "Actual execution still requires rerunning Step 149 on H200 with --expect-h200 --min-gpu-count 1",
        {"required_args": "--expect-h200 --min-gpu-count 1"},
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
    }

    for key, value in lock_values.items():
        add_check(
            checks,
            f"lock_{key}_false",
            value is False,
            "hard",
            f"{key} must remain false in Step 151-A draft",
            {"value": value},
        )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    release_manifest_status = STEP151A_STATUS if hard_failures == 0 else "BLOCKED"

    manifest: Dict[str, Any] = {
        "artifact_version": "explicit_operator_release_manifest_draft_step151a_v1",
        "step": "151-A",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "input_step": 150,
        "step150_manifest": str(step150_manifest_path),
        "step150_checklist_status": step150_status,
        "release_manifest_status": release_manifest_status,
        "decision": STEP151A_DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "operator_approval_recorded": False,
        "h200_expected_preflight_passed": False,
        "h200_expected_preflight_required": True,
        "h200_expected_preflight_required_args": "--expect-h200 --min-gpu-count 1",
        "release_scope": {
            "target_matrix": "A/A90/A80/A70 x R0-R5 x seeds 1-3",
            "target_run_count": 72,
            "baselines_required": ["B0R", "B1", "B2"],
            "release_type": "draft_only_still_locked",
        },
        "required_before_actual_release": [
            "operator_approval_recorded_true",
            "real_h200_step149_expect_h200_pass",
            "cuda_available_true_on_h200",
            "gpu_count_at_least_1_on_h200",
            "git_commit_pinned",
            "worktree_clean_or_dirty_diff_recorded",
            "output_root_empty_or_archived",
            "actual_execution_release_manifest_separate_from_draft",
        ],
        "next_gate": NEXT_GATE,
        "notes": [
            "Step 151-A is a release manifest draft only.",
            "It intentionally keeps actual execution locked.",
            "A separate later release step is required before any actual reward ablation run.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "explicit_operator_release_manifest_draft_step151a_manifest.json"
    summary_path = output_root / "explicit_operator_release_manifest_draft_step151a_summary.md"

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
    parser.add_argument("--step150-manifest", default=DEFAULT_STEP150_MANIFEST)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/explicit_operator_release_manifest_draft_step151a",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    step150_manifest_path = Path(args.step150_manifest)
    if not step150_manifest_path.is_absolute():
        step150_manifest_path = project_root / step150_manifest_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    manifest = build_manifest(
        project_root=project_root,
        step150_manifest_path=step150_manifest_path,
        output_root=output_root,
    )

    print("[OK] Step 151-A explicit operator release manifest draft completed")
    print(f"[OK] release_manifest_status : {manifest['release_manifest_status']}")
    print(f"[OK] decision                : {manifest['decision']}")
    print(f"[OK] hard_failures           : {manifest['hard_failures']}")
    print(f"[OK] warnings                : {manifest['warnings']}")
    print(f"[OK] actual_execution_allowed  : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed             : {manifest['train_allowed']}")
    print(f"[OK] manifest                : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
