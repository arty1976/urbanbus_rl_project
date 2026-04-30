from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_151A_STATUS = "RELEASE_MANIFEST_DRAFT_STILL_LOCKED"
EXPECTED_151B_STATUS = "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED"
ACCEPTED_151C_STATUSES = {
    "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED",
    "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED",
}
EXPECTED_151D_STATUS = "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED"

EXPECTED_151A_DECISION = "DRAFT_ONLY_NOT_RELEASED"
EXPECTED_151B_DECISION = "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED"
EXPECTED_151C_DECISION = "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED"
EXPECTED_151D_DECISION = "APPROVAL_DRAFT_ONLY_NOT_APPROVED_NOT_RELEASED"

STEP151E_STATUS = "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED"
STEP151E_DECISION = "INDEX_ONLY_HANDOFF_NOT_RELEASED"
NEXT_GATE = "READY_FOR_H200_TRANSFER_OR_REAL_H200_STEP149_EXPECT_H200_RERUN"

DEFAULT_STEP151A_MANIFEST = (
    "artifacts/rewards/explicit_operator_release_manifest_draft_step151a/"
    "explicit_operator_release_manifest_draft_step151a_manifest.json"
)
DEFAULT_STEP151B_MANIFEST = (
    "artifacts/rewards/h200_expected_preflight_rerun_checklist_step151b/"
    "h200_expected_preflight_rerun_checklist_step151b_manifest.json"
)
DEFAULT_STEP151C_MANIFEST = (
    "artifacts/rewards/h200_actual_preflight_rerun_result_intake_placeholder_step151c/"
    "h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
)
DEFAULT_STEP151D_MANIFEST = (
    "artifacts/rewards/operator_approval_decision_draft_step151d/"
    "operator_approval_decision_draft_step151d_manifest.json"
)

H200_STEP149_COMMAND = (
    "python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py "
    "--project-root /workspace/urbanbus_rl_project "
    "--expect-h200 "
    "--min-gpu-count 1 "
    "--output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual"
)

H200_STEP149_EXPECTED_MANIFEST = (
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


def path_arg(project_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    return p


def read_required_manifest(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return load_json(path)


def validate_lock_fields(payload: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    values = {}
    passed = True
    for field in fields:
        value = coerce_bool(payload.get(field), default=None)
        values[field] = value
        if value is not False:
            passed = False
    return {"passed": passed, "values": values}


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Step 151-E H200 handoff packet index summary",
        "",
        f"- handoff_index_status: {manifest.get('handoff_index_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
        f"- h200_expected_preflight_passed: {manifest.get('h200_expected_preflight_passed')}",
        f"- operator_approval_recorded: {manifest.get('operator_approval_recorded')}",
        f"- actual_execution_allowed: {manifest.get('actual_execution_allowed')}",
        f"- actual_execution_released: {manifest.get('actual_execution_released')}",
        f"- train_allowed: {manifest.get('train_allowed')}",
        f"- next_gate: {manifest.get('next_gate')}",
        "",
        "## H200 command",
        "",
        "```bash",
        manifest.get("h200_step149_command", ""),
        "```",
        "",
        "## Handoff packet manifests",
        "",
    ]

    for item in manifest.get("handoff_packet_manifests", []):
        lines.append(f"- {item.get('label')}: `{item.get('path')}`")

    lines.extend(["", "## Checks", ""])

    for check in manifest.get("checks", []):
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(
            f"- [{mark}] {check.get('check_id')} | {check.get('severity')} | {check.get('message')}"
        )

    lines.extend(
        [
            "",
            "## Lock statement",
            "",
            "Step 151-E is an index-only handoff packet. It does not approve, release, or run actual reward ablation execution.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(
    project_root: Path,
    step151a_manifest_path: Path,
    step151b_manifest_path: Path,
    step151c_manifest_path: Path,
    step151d_manifest_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    step151a = read_required_manifest(step151a_manifest_path, "Step 151-A manifest")
    step151b = read_required_manifest(step151b_manifest_path, "Step 151-B manifest")
    step151c = read_required_manifest(step151c_manifest_path, "Step 151-C manifest")
    step151d = read_required_manifest(step151d_manifest_path, "Step 151-D manifest")

    checks: List[Dict[str, Any]] = []

    status_a = str(step151a.get("release_manifest_status", ""))
    status_b = str(step151b.get("checklist_status", ""))
    status_c = str(step151c.get("intake_status", ""))
    status_d = str(step151d.get("approval_decision_status", ""))

    decision_a = str(step151a.get("decision", ""))
    decision_b = str(step151b.get("decision", ""))
    decision_c = str(step151c.get("decision", ""))
    decision_d = str(step151d.get("decision", ""))

    add_check(
        checks,
        "step151a_status_and_decision_valid",
        status_a == EXPECTED_151A_STATUS and decision_a == EXPECTED_151A_DECISION and coerce_int(step151a.get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-A must be draft-only, still locked, and hard_failures=0",
        {"status": status_a, "decision": decision_a, "path": str(step151a_manifest_path)},
    )

    add_check(
        checks,
        "step151b_status_and_decision_valid",
        status_b == EXPECTED_151B_STATUS and decision_b == EXPECTED_151B_DECISION and coerce_int(step151b.get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-B must be H200 rerun checklist-only, still locked, and hard_failures=0",
        {"status": status_b, "decision": decision_b, "path": str(step151b_manifest_path)},
    )

    add_check(
        checks,
        "step151c_status_and_decision_valid",
        status_c in ACCEPTED_151C_STATUSES and decision_c == EXPECTED_151C_DECISION and coerce_int(step151c.get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-C must be waiting or received-and-validated, still locked, and hard_failures=0",
        {"status": status_c, "decision": decision_c, "path": str(step151c_manifest_path)},
    )

    add_check(
        checks,
        "step151d_status_and_decision_valid",
        status_d == EXPECTED_151D_STATUS and decision_d == EXPECTED_151D_DECISION and coerce_int(step151d.get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-D must be approval decision draft-only, still locked, and hard_failures=0",
        {"status": status_d, "decision": decision_d, "path": str(step151d_manifest_path)},
    )

    lock_fields = [
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]

    for label, payload in [
        ("step151a", step151a),
        ("step151b", step151b),
        ("step151c", step151c),
        ("step151d", step151d),
    ]:
        result = validate_lock_fields(payload, lock_fields)
        add_check(
            checks,
            f"{label}_lock_fields_false",
            result["passed"],
            "hard",
            f"{label} must keep all execution/result/claim locks false",
            result["values"],
        )

    approval_lock = validate_lock_fields(
        step151d,
        ["operator_approval_recorded", "operator_approval_granted", "operator_approval_decision_final"],
    )
    add_check(
        checks,
        "step151d_operator_approval_locks_false",
        approval_lock["passed"],
        "hard",
        "Step 151-D must not record or grant operator approval",
        approval_lock["values"],
    )

    h200_passed = coerce_bool(step151c.get("h200_expected_preflight_passed"), default=False)
    h200_present = coerce_bool(step151c.get("h200_expected_preflight_manifest_present"), default=False)

    add_check(
        checks,
        "h200_step149_command_recorded",
        True,
        "hard",
        "H200 Step 149 rerun command is recorded in the handoff index",
        {"command": H200_STEP149_COMMAND},
    )

    add_check(
        checks,
        "handoff_index_is_not_release",
        True,
        "hard",
        "Step 151-E is an index-only handoff packet and cannot release actual execution",
        {"decision": STEP151E_DECISION},
    )

    git_status_short = get_git_status_short(project_root)
    add_check(
        checks,
        "git_status_recorded",
        git_status_short != "UNKNOWN",
        "warning",
        "Git worktree status is recorded for handoff review",
        {"git_status_short": git_status_short},
    )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    status = STEP151E_STATUS if hard_failures == 0 else "BLOCKED"

    if hard_failures != 0:
        next_gate = "BLOCKED_UNTIL_STEP151_HANDOFF_INPUTS_FIXED"
    elif h200_passed is True:
        next_gate = "READY_FOR_SEPARATE_OPERATOR_APPROVAL_RECORD_OR_RELEASE_REVIEW"
    else:
        next_gate = NEXT_GATE

    handoff_packet_manifests = [
        {"label": "step151a_release_manifest_draft", "path": str(step151a_manifest_path)},
        {"label": "step151b_h200_expected_preflight_rerun_checklist", "path": str(step151b_manifest_path)},
        {"label": "step151c_h200_result_intake_placeholder", "path": str(step151c_manifest_path)},
        {"label": "step151d_operator_approval_decision_draft", "path": str(step151d_manifest_path)},
    ]

    tracked_handoff_files = [
        "step151a_explicit_operator_release_manifest_draft_still_locked.ps1",
        "step151b_h200_expected_preflight_rerun_checklist_still_locked.ps1",
        "step151c_h200_actual_preflight_rerun_result_intake_placeholder_still_locked.ps1",
        "step151d_operator_approval_decision_draft_still_locked.ps1",
        "05_training/rewards/h200_environment_preflight_result_manifest_step149.py",
        "05_training/rewards/validate_h200_environment_preflight_result_manifest_step149.py",
    ]

    manifest: Dict[str, Any] = {
        "artifact_version": "h200_handoff_packet_index_step151e_v1",
        "step": "151-E",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "handoff_index_status": status,
        "decision": STEP151E_DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "input_steps": ["151-A", "151-B", "151-C", "151-D"],
        "handoff_packet_manifests": handoff_packet_manifests,
        "tracked_handoff_files": tracked_handoff_files,
        "h200_step149_command": H200_STEP149_COMMAND,
        "h200_step149_expected_manifest": H200_STEP149_EXPECTED_MANIFEST,
        "h200_expected_preflight_manifest_present": bool(h200_present),
        "h200_expected_preflight_passed": bool(h200_passed),
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
        "handoff_packet_index_only": True,
        "actual_release_requires_separate_manifest": True,
        "recommended_h200_sequence": [
            "Transfer tracked source/spec/validator files to /workspace/urbanbus_rl_project",
            "Verify git commit or record dirty diff on H200",
            "Run Step 149 with --expect-h200 --min-gpu-count 1",
            "Copy or sync the H200 Step 149 result manifest to the expected path",
            "Rerun Step 151-C intake validation with the H200 result manifest present",
            "Rerun Step 151-D approval decision draft",
            "Create a separate explicit actual execution release manifest only after approval is recorded",
        ],
        "required_before_actual_release": [
            "real_h200_step149_expect_h200_pass",
            "h200_expected_preflight_manifest_present_true",
            "h200_expected_preflight_passed_true",
            "operator_approval_recorded_true",
            "operator_approval_granted_true",
            "separate_actual_execution_release_manifest",
            "output_root_empty_or_archived",
            "git_commit_pinned",
        ],
        "next_gate": next_gate,
        "notes": [
            "Step 151-E is a handoff index only.",
            "It intentionally does not run H200 preflight.",
            "It intentionally does not record operator approval.",
            "It intentionally does not release actual reward ablation execution.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "h200_handoff_packet_index_step151e_manifest.json"
    summary_path = output_root / "h200_handoff_packet_index_step151e_summary.md"

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
    parser.add_argument("--step151b-manifest", default=DEFAULT_STEP151B_MANIFEST)
    parser.add_argument("--step151c-manifest", default=DEFAULT_STEP151C_MANIFEST)
    parser.add_argument("--step151d-manifest", default=DEFAULT_STEP151D_MANIFEST)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_handoff_packet_index_step151e",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    manifest = build_manifest(
        project_root=project_root,
        step151a_manifest_path=path_arg(project_root, args.step151a_manifest),
        step151b_manifest_path=path_arg(project_root, args.step151b_manifest),
        step151c_manifest_path=path_arg(project_root, args.step151c_manifest),
        step151d_manifest_path=path_arg(project_root, args.step151d_manifest),
        output_root=path_arg(project_root, args.output_root),
    )

    print("[OK] Step 151-E H200 handoff packet index completed")
    print(f"[OK] handoff_index_status : {manifest['handoff_index_status']}")
    print(f"[OK] decision             : {manifest['decision']}")
    print(f"[OK] hard_failures        : {manifest['hard_failures']}")
    print(f"[OK] warnings             : {manifest['warnings']}")
    print(f"[OK] h200_expected_preflight_passed : {manifest['h200_expected_preflight_passed']}")
    print(f"[OK] operator_approval_recorded     : {manifest['operator_approval_recorded']}")
    print(f"[OK] actual_execution_allowed       : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released      : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed                  : {manifest['train_allowed']}")
    print(f"[OK] manifest            : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
