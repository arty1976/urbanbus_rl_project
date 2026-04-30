from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_STEP150_STATUS = "READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT"
EXPECTED_STEP151A_STATUS = "RELEASE_MANIFEST_DRAFT_STILL_LOCKED"
EXPECTED_STEP151B_STATUS = "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED"
ACCEPTED_STEP151C_STATUSES = {
    "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED",
    "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED",
}
EXPECTED_STEP151D_STATUS = "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED"
EXPECTED_STEP151E_STATUS = "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED"

STEP151F_STATUS_UPDATED = "PROJECT_LOG_UPDATED"
STEP151F_STATUS_ALREADY_PRESENT = "PROJECT_LOG_ALREADY_PRESENT"
STEP151F_DECISION = "PROJECT_LOG_ONLY_EXECUTION_STILL_LOCKED"
LOG_MARKER = "Step 151-F project log update: Step 150~151-E H200 handoff guard chain"

DEFAULTS = {
    "step150": "artifacts/rewards/actual_reward_ablation_operator_release_checklist_step150/actual_reward_ablation_operator_release_checklist_step150_manifest.json",
    "step151a": "artifacts/rewards/explicit_operator_release_manifest_draft_step151a/explicit_operator_release_manifest_draft_step151a_manifest.json",
    "step151b": "artifacts/rewards/h200_expected_preflight_rerun_checklist_step151b/h200_expected_preflight_rerun_checklist_step151b_manifest.json",
    "step151c": "artifacts/rewards/h200_actual_preflight_rerun_result_intake_placeholder_step151c/h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json",
    "step151d": "artifacts/rewards/operator_approval_decision_draft_step151d/operator_approval_decision_draft_step151d_manifest.json",
    "step151e": "artifacts/rewards/h200_handoff_packet_index_step151e/h200_handoff_packet_index_step151e_manifest.json",
}

LOCK_FIELDS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

APPROVAL_LOCK_FIELDS = [
    "operator_approval_recorded",
    "operator_approval_granted",
]


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


def read_text_any(path: Path) -> str:
    if not path.exists():
        return "# 🚌 UrbanBus RL Project - 작업 일지 (Project Journal)\n\n"
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read text: {path}")


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


def path_arg(project_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    return p


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


def require_manifest(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return load_json(path)


def lock_check(payload: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    passed = True
    for field in fields:
        value = coerce_bool(payload.get(field), default=None)
        values[field] = value
        if value is not False:
            passed = False
    return {"passed": passed, "values": values}


def extract_statuses(manifests: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "step150_checklist_status": manifests["step150"].get("checklist_status"),
        "step151a_release_manifest_status": manifests["step151a"].get("release_manifest_status"),
        "step151b_checklist_status": manifests["step151b"].get("checklist_status"),
        "step151c_intake_status": manifests["step151c"].get("intake_status"),
        "step151d_approval_decision_status": manifests["step151d"].get("approval_decision_status"),
        "step151e_handoff_index_status": manifests["step151e"].get("handoff_index_status"),
        "h200_expected_preflight_manifest_present": manifests["step151c"].get("h200_expected_preflight_manifest_present"),
        "h200_expected_preflight_passed": manifests["step151c"].get("h200_expected_preflight_passed"),
    }


def build_log_section(statuses: Dict[str, Any], created_at_utc: str) -> str:
    h200_present = statuses.get("h200_expected_preflight_manifest_present")
    h200_passed = statuses.get("h200_expected_preflight_passed")

    return f"""
---
## 📅 2026-04-30
### {LOG_MARKER}

Step 150부터 Step 151-E까지의 actual reward ablation release-preflight guard chain을 기록했다. 이 구간의 목적은 A/A90/A80/A70 × R0~R5 × seeds 1,2,3 = 72-run reward ablation 실행을 바로 여는 것이 아니라, 실제 H200 서버에서 실행 전 반드시 확인해야 할 operator release / H200 preflight / approval / handoff 경로를 단계적으로 잠금 상태로 고정하는 것이다.

#### 1. 완료된 guard chain

| Step | 상태 | 의미 |
|---|---|---|
| Step 150 | `{statuses.get('step150_checklist_status')}` | actual reward ablation operator checklist 통과 |
| Step 151-A | `{statuses.get('step151a_release_manifest_status')}` | explicit operator release manifest draft 생성, still locked |
| Step 151-B | `{statuses.get('step151b_checklist_status')}` | H200 Step 149 `--expect-h200 --min-gpu-count 1` 재실행 checklist 생성, still locked |
| Step 151-C | `{statuses.get('step151c_intake_status')}` | H200 Step 149 실제 결과 manifest intake 틀 생성 |
| Step 151-D | `{statuses.get('step151d_approval_decision_status')}` | operator approval decision draft 생성, approval 미기록 |
| Step 151-E | `{statuses.get('step151e_handoff_index_status')}` | H200 handoff packet index 생성 |

#### 2. 현재 H200 상태

- h200_expected_preflight_manifest_present = `{h200_present}`
- h200_expected_preflight_passed = `{h200_passed}`
- 현재 로컬 기준 다음 외부 작업은 실제 H200 서버에서 Step 149를 아래 조건으로 재실행하는 것이다.

```text
--expect-h200 --min-gpu-count 1
```

#### 3. 유지되는 방어선

아래 값은 모두 false로 유지된다.

```text
actual_execution_allowed = false
actual_execution_released = false
train_allowed = false
actual_results = false
winner_selected = false
trainable_reward_promoted = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
operator_approval_recorded = false
operator_approval_granted = false
```

#### 4. 연구적 의미

이 단계까지의 산출물은 실제 성능 결과가 아니다. actual reward ablation 실행도 아직 열리지 않았다. 다만 H200 서버로 넘길 preflight packet index가 만들어졌고, 다음에 실제 H200 환경에서 Step 149를 재실행한 결과 manifest가 들어오면 Step 151-C intake를 재검증할 수 있는 구조가 준비되었다.

#### 5. 기록 메타데이터

- created_at_utc = `{created_at_utc}`
- documentation_only = true
- actual_execution_released = false
""".strip() + "\n"


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Step 151-F project log update summary",
        "",
        f"- log_update_status: {manifest.get('log_update_status')}",
        f"- decision: {manifest.get('decision')}",
        f"- hard_failures: {manifest.get('hard_failures')}",
        f"- warnings: {manifest.get('warnings')}",
        f"- project_log: {manifest.get('project_log')}",
        f"- project_log_contains_marker: {manifest.get('project_log_contains_marker')}",
        f"- actual_execution_allowed: {manifest.get('actual_execution_allowed')}",
        f"- actual_execution_released: {manifest.get('actual_execution_released')}",
        f"- train_allowed: {manifest.get('train_allowed')}",
        "",
        "## Checks",
        "",
    ]
    for check in manifest.get("checks", []):
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- [{mark}] {check.get('check_id')} | {check.get('severity')} | {check.get('message')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_update(
    project_root: Path,
    project_log: Path,
    output_root: Path,
    manifest_paths: Dict[str, Path],
) -> Dict[str, Any]:
    manifests = {
        "step150": require_manifest(manifest_paths["step150"], "Step 150 manifest"),
        "step151a": require_manifest(manifest_paths["step151a"], "Step 151-A manifest"),
        "step151b": require_manifest(manifest_paths["step151b"], "Step 151-B manifest"),
        "step151c": require_manifest(manifest_paths["step151c"], "Step 151-C manifest"),
        "step151d": require_manifest(manifest_paths["step151d"], "Step 151-D manifest"),
        "step151e": require_manifest(manifest_paths["step151e"], "Step 151-E manifest"),
    }

    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "step150_status_ready",
        manifests["step150"].get("checklist_status") == EXPECTED_STEP150_STATUS and coerce_int(manifests["step150"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 150 checklist must be ready for Step 151 and hard_failures=0",
        {"path": str(manifest_paths["step150"]), "status": manifests["step150"].get("checklist_status")},
    )
    add_check(
        checks,
        "step151a_status_ready",
        manifests["step151a"].get("release_manifest_status") == EXPECTED_STEP151A_STATUS and coerce_int(manifests["step151a"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-A must be still-locked release draft and hard_failures=0",
        {"path": str(manifest_paths["step151a"]), "status": manifests["step151a"].get("release_manifest_status")},
    )
    add_check(
        checks,
        "step151b_status_ready",
        manifests["step151b"].get("checklist_status") == EXPECTED_STEP151B_STATUS and coerce_int(manifests["step151b"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-B must be H200 expected preflight rerun checklist and hard_failures=0",
        {"path": str(manifest_paths["step151b"]), "status": manifests["step151b"].get("checklist_status")},
    )
    add_check(
        checks,
        "step151c_status_accepted",
        manifests["step151c"].get("intake_status") in ACCEPTED_STEP151C_STATUSES and coerce_int(manifests["step151c"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-C must be waiting or received-and-validated and hard_failures=0",
        {"path": str(manifest_paths["step151c"]), "status": manifests["step151c"].get("intake_status")},
    )
    add_check(
        checks,
        "step151d_status_ready",
        manifests["step151d"].get("approval_decision_status") == EXPECTED_STEP151D_STATUS and coerce_int(manifests["step151d"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-D must be operator approval decision draft and hard_failures=0",
        {"path": str(manifest_paths["step151d"]), "status": manifests["step151d"].get("approval_decision_status")},
    )
    add_check(
        checks,
        "step151e_status_ready",
        manifests["step151e"].get("handoff_index_status") == EXPECTED_STEP151E_STATUS and coerce_int(manifests["step151e"].get("hard_failures"), 999) == 0,
        "hard",
        "Step 151-E must be H200 handoff packet index and hard_failures=0",
        {"path": str(manifest_paths["step151e"]), "status": manifests["step151e"].get("handoff_index_status")},
    )

    for label, payload in manifests.items():
        lock = lock_check(payload, LOCK_FIELDS)
        add_check(
            checks,
            f"{label}_execution_result_claim_locks_false",
            lock["passed"],
            "hard",
            f"{label} must keep execution/result/claim locks false",
            lock["values"],
        )

    approval_lock_payloads = {
        "step151d": manifests["step151d"],
        "step151e": manifests["step151e"],
    }
    for label, payload in approval_lock_payloads.items():
        lock = lock_check(payload, APPROVAL_LOCK_FIELDS)
        add_check(
            checks,
            f"{label}_operator_approval_locks_false",
            lock["passed"],
            "hard",
            f"{label} must keep operator approval locks false",
            lock["values"],
        )

    git_status_short = get_git_status_short(project_root)
    add_check(
        checks,
        "git_status_recorded",
        git_status_short != "UNKNOWN",
        "warning",
        "Git worktree status is recorded in Step 151-F manifest",
        {"git_status_short": git_status_short},
    )

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    statuses = extract_statuses(manifests)
    created_at = utc_now()

    project_log.parent.mkdir(parents=True, exist_ok=True)
    before = read_text_any(project_log)

    if LOG_MARKER in before:
        log_update_status = STEP151F_STATUS_ALREADY_PRESENT
        after = before
        action = "marker_already_present_no_duplicate_append"
    else:
        section = build_log_section(statuses, created_at)
        after = before.rstrip() + "\n\n" + section + "\n"
        project_log.write_text(after, encoding="utf-8")
        log_update_status = STEP151F_STATUS_UPDATED
        action = "appended_step151f_section"

    project_log_contains_marker = LOG_MARKER in after
    add_check(
        checks,
        "project_log_contains_step151f_marker",
        project_log_contains_marker,
        "hard",
        "project_log.md must contain the Step 151-F marker after update",
        {"project_log": str(project_log), "marker": LOG_MARKER},
    )

    # Recompute hard failures after marker check.
    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warnings = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])

    manifest: Dict[str, Any] = {
        "artifact_version": "project_log_update_step151f_v1",
        "step": "151-F",
        "created_at_utc": created_at,
        "project_root": str(project_root),
        "git_commit": get_git_commit(project_root),
        "git_status_short": git_status_short,
        "log_update_status": log_update_status if hard_failures == 0 else "BLOCKED",
        "decision": STEP151F_DECISION,
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "checks": checks,
        "project_log": str(project_log),
        "project_log_contains_marker": project_log_contains_marker,
        "action": action,
        "input_manifests": {k: str(v) for k, v in manifest_paths.items()},
        "status_snapshot": statuses,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
        "documentation_only": True,
        "actual_release_requires_separate_manifest": True,
        "next_gate": "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RERUN_OR_STEP152_HANDOFF_PACKAGE_EXPORT",
        "notes": [
            "Step 151-F only updates project_log.md.",
            "Actual reward ablation execution remains locked.",
            "The next external action is real H200 Step 149 rerun with --expect-h200 --min-gpu-count 1.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "project_log_update_step151f_manifest.json"
    summary_path = output_root / "project_log_update_step151f_summary.md"
    manifest["output_files"] = {"manifest": str(manifest_path), "summary": str(summary_path)}
    dump_json(manifest_path, manifest)
    write_summary(summary_path, manifest)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--project-log", default="project_log.md")
    parser.add_argument("--step150-manifest", default=DEFAULTS["step150"])
    parser.add_argument("--step151a-manifest", default=DEFAULTS["step151a"])
    parser.add_argument("--step151b-manifest", default=DEFAULTS["step151b"])
    parser.add_argument("--step151c-manifest", default=DEFAULTS["step151c"])
    parser.add_argument("--step151d-manifest", default=DEFAULTS["step151d"])
    parser.add_argument("--step151e-manifest", default=DEFAULTS["step151e"])
    parser.add_argument("--output-root", default="artifacts/rewards/project_log_update_step151f")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    project_log = path_arg(project_root, args.project_log)
    output_root = path_arg(project_root, args.output_root)

    manifest_paths = {
        "step150": path_arg(project_root, args.step150_manifest),
        "step151a": path_arg(project_root, args.step151a_manifest),
        "step151b": path_arg(project_root, args.step151b_manifest),
        "step151c": path_arg(project_root, args.step151c_manifest),
        "step151d": path_arg(project_root, args.step151d_manifest),
        "step151e": path_arg(project_root, args.step151e_manifest),
    }

    manifest = build_update(
        project_root=project_root,
        project_log=project_log,
        output_root=output_root,
        manifest_paths=manifest_paths,
    )

    print("[OK] Step 151-F project log update completed")
    print(f"[OK] log_update_status : {manifest['log_update_status']}")
    print(f"[OK] decision          : {manifest['decision']}")
    print(f"[OK] hard_failures     : {manifest['hard_failures']}")
    print(f"[OK] warnings          : {manifest['warnings']}")
    print(f"[OK] project_log       : {manifest['project_log']}")
    print(f"[OK] action            : {manifest['action']}")
    print(f"[OK] actual_execution_allowed  : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed             : {manifest['train_allowed']}")
    print(f"[OK] manifest         : {manifest['output_files']['manifest']}")

    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
