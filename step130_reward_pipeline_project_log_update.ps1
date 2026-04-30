$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 130 status summary JSON
# ---------------------------------------------------------------------
$StatusJson = @'
{
  "artifact_version": "reward_pipeline_project_log_update_step130_v1",
  "step": 130,
  "log_update_status": "PROJECT_LOG_UPDATE_READY",
  "purpose": "Record Step 111 through Step 129 reward pipeline status in project_log.md and preserve the no-promotion/no-training guard state.",
  "covered_steps": [
    "Step111 final reward specification draft",
    "Step112 reward candidate protocol",
    "Step113 reward normalization baseline lock",
    "Step114 hard constraint review",
    "Step115 reward ablation matrix",
    "Step116 trainable reward promotion gate",
    "Step117 reward ablation result schema",
    "Step118 reward ablation result writer guard",
    "Step119 reward ablation runner dry-run plan",
    "Step120 reward ablation execution preflight",
    "Step121 reward ablation execution manifest",
    "Step122 reward ablation sandbox no-op runner guard",
    "Step123 reward ablation no-op result ingestion guard",
    "Step124 reward ablation actual result schema bridge",
    "Step125 reward ablation selection criteria gate",
    "Step126 reward ablation actual result ingestion preflight",
    "Step127 trainable reward promotion decision package",
    "Step128 reward pipeline actual ablation wait state",
    "Step129 actual reward ablation runbook"
  ],
  "reward_pipeline_current_state": {
    "reward_spec_status": "draft_not_trainable",
    "candidate_matrix": "R0_to_R5_defined",
    "normalization_reference": "B1_noop_reference_name_locked_numeric_values_not_locked",
    "hard_constraints": "reviewed_not_training_unlock",
    "dry_run_plan_rows": 72,
    "execution_manifest_rows": 72,
    "sandbox_runner": "noop_only",
    "actual_result_ingestion": "not_ingested",
    "promotion_decision": "not_promoted",
    "runbook": "ready_not_executed"
  },
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "expected_actual_ablation_runs": 72,
  "guard_flags": {
    "actual_execution_started": false,
    "actual_results": false,
    "actual_ablation_data_available": false,
    "winner_selected": false,
    "trainable_reward_promoted": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed": false,
    "final_reward_design_claim_allowed": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "next_recommended_status": "READY_FOR_STEP131_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT_OR_PUSH_SYNC"
}
'@

$StatusJsonPath = ".\05_training\rewards\reward_pipeline_project_log_update_step130.json"
$StatusJson | Set-Content -Path $StatusJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 130 append/update project_log.md script
# ---------------------------------------------------------------------
$UpdaterPy = @'
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SECTION_MARKER = "<!-- STEP130_REWARD_PIPELINE_STATUS_START -->"
SECTION_END = "<!-- STEP130_REWARD_PIPELINE_STATUS_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def build_section(status: Dict[str, Any]) -> str:
    covered_lines = "\n".join(f"- {item}" for item in status["covered_steps"])
    guard_lines = "\n".join(
        f"- `{key}` = `{str(value).lower()}`"
        for key, value in status["guard_flags"].items()
    )

    return f"""{SECTION_MARKER}
## Step 130 — Reward pipeline status update

작성 시각(UTC): {utc_now()}

### 목적

Step 111부터 Step 129까지 이어진 reward 설계·검증·실행준비 pipeline을 project log에 고정한다. 이 기록은 reward 후보군이 정의되었더라도 아직 실제 reward ablation 결과가 없고, 따라서 winner 선택·reward 승격·MAPPO 학습 허용이 모두 금지 상태임을 명확히 남기기 위한 것이다.

### 포함된 단계

{covered_lines}

### 현재 reward pipeline 상태

- reward spec: `{status["reward_pipeline_current_state"]["reward_spec_status"]}`
- candidate matrix: `{status["reward_pipeline_current_state"]["candidate_matrix"]}`
- normalization reference: `{status["reward_pipeline_current_state"]["normalization_reference"]}`
- hard constraints: `{status["reward_pipeline_current_state"]["hard_constraints"]}`
- dry-run plan rows: `{status["reward_pipeline_current_state"]["dry_run_plan_rows"]}`
- execution manifest rows: `{status["reward_pipeline_current_state"]["execution_manifest_rows"]}`
- sandbox runner: `{status["reward_pipeline_current_state"]["sandbox_runner"]}`
- actual result ingestion: `{status["reward_pipeline_current_state"]["actual_result_ingestion"]}`
- promotion decision: `{status["reward_pipeline_current_state"]["promotion_decision"]}`
- runbook: `{status["reward_pipeline_current_state"]["runbook"]}`

### Reward ablation matrix

- candidates: `{", ".join(status["candidate_ids"])}`
- conditions: `{", ".join(status["condition_ids"])}`
- seeds: `{", ".join(str(x) for x in status["seeds"])}`
- expected actual ablation runs: `{status["expected_actual_ablation_runs"]}`

### 유지되는 guard flags

{guard_lines}

### 해석

현재까지의 작업은 reward 후보 설계, 후보군 비교 계획, 결과 schema, dry-run plan, execution manifest, no-op guard, actual-result ingestion preflight, promotion decision package, actual ablation runbook을 준비한 것이다. 하지만 실제 reward ablation 결과는 아직 존재하지 않는다.

따라서 다음 주장은 모두 금지된다.

- R0~R5 중 특정 reward가 최고라는 주장
- trainable reward가 승격되었다는 주장
- 현재 reward 후보로 MAPPO 학습을 시작해도 된다는 주장
- 논문 수준 성능 개선 주장
- causal performance claim

### 다음 상태

`{status["next_recommended_status"]}`

{SECTION_END}
"""


def update_project_log(project_log: Path, status: Dict[str, Any]) -> Dict[str, Any]:
    project_log.parent.mkdir(parents=True, exist_ok=True)
    if project_log.exists():
        text = project_log.read_text(encoding="utf-8-sig")
    else:
        text = "# urbanbus_rl_project project log\n\n"

    new_section = build_section(status)

    if SECTION_MARKER in text and SECTION_END in text:
        before = text.split(SECTION_MARKER)[0].rstrip()
        after = text.split(SECTION_END, 1)[1].lstrip()
        updated = before + "\n\n" + new_section + "\n\n" + after
        action = "replaced_existing_step130_section"
    else:
        updated = text.rstrip() + "\n\n" + new_section + "\n"
        action = "appended_step130_section"

    project_log.write_text(updated, encoding="utf-8")
    return {
        "project_log_path": str(project_log),
        "action": action,
        "section_marker": SECTION_MARKER,
        "section_end": SECTION_END,
        "updated": True,
    }


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="05_training/rewards/reward_pipeline_project_log_update_step130.json")
    parser.add_argument("--project-log", default="project_log.md")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_pipeline_project_log_update_step130")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    status_path = resolve_path(project_root, args.status)
    project_log = resolve_path(project_root, args.project_log)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    status = load_json_any_encoding(status_path)
    update_result = update_project_log(project_log, status)

    manifest = {
        "artifact_version": "reward_pipeline_project_log_update_step130_manifest_v1",
        "created_at_utc": utc_now(),
        "log_update_status": "PROJECT_LOG_UPDATED",
        "project_log_path": str(project_log),
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "update_result": update_result,
        "source_status_snapshot": status,
    }

    manifest_path = output_root / "reward_pipeline_project_log_update_step130_manifest.json"
    dump_json(manifest_path, manifest)

    print("[OK] Step 130 reward pipeline project log update completed")
    print(f"[OK] log_update_status: {manifest['log_update_status']}")
    print(f"[OK] project_log     : {project_log}")
    print(f"[OK] action          : {update_result['action']}")
    print(f"[OK] actual_results  : {manifest['actual_results']}")
    print(f"[OK] winner_selected : {manifest['winner_selected']}")
    print(f"[OK] promoted        : {manifest['trainable_reward_promoted']}")
    print(f"[OK] train_allowed   : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest        : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$UpdaterPath = ".\05_training\rewards\update_project_log_reward_pipeline_step130.py"
$UpdaterPy | Set-Content -Path $UpdaterPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 130 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SECTION_MARKER = "<!-- STEP130_REWARD_PIPELINE_STATUS_START -->"
SECTION_END = "<!-- STEP130_REWARD_PIPELINE_STATUS_END -->"
EXPECTED_STEPS = list(range(111, 130))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
        "severity": "blocker",
        "blocking": True,
    })


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    project_log = Path(manifest.get("project_log_path", ""))
    add_check(checks, "log_update_status", "PROJECT_LOG_UPDATED", manifest.get("log_update_status"), "Project log must be updated.")
    add_check(checks, "project_log_exists", True, project_log.exists(), "project_log.md must exist.")

    text = project_log.read_text(encoding="utf-8-sig") if project_log.exists() else ""
    add_check(checks, "section_marker_present", True, SECTION_MARKER in text, "Step 130 marker must exist.")
    add_check(checks, "section_end_present", True, SECTION_END in text, "Step 130 end marker must exist.")

    for step in EXPECTED_STEPS:
        found = f"Step{step}" in text or f"Step {step}" in text
        add_check(checks, f"log_mentions_step_{step}", True, found, f"project_log must mention Step {step}.")

    required_phrases = [
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "실제 reward ablation 결과는 아직 존재하지 않는다",
    ]
    for phrase in required_phrases:
        add_check(checks, f"phrase_{phrase[:24]}", True, phrase in text, f"project_log must include phrase: {phrase}")

    for key in [
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        add_check(checks, f"manifest_guard_{key}", False, bool(manifest.get(key)), f"{key} must remain false.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_pipeline_project_log_update_step130_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_PROJECT_LOG_UPDATED_REWARD_PIPELINE_WAITING"
            if audit_status == "PASS"
            else "FAIL_PROJECT_LOG_UPDATE"
        ),
        "next_status": (
            "READY_FOR_STEP131_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT_OR_PUSH_SYNC"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP130_PROJECT_LOG"
        ),
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "failure_count": len(failures),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_pipeline_project_log_update_step130_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json_any_encoding(manifest_path)
    report = validate_manifest(manifest)
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_pipeline_project_log_update_step130_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 130 reward pipeline project log validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] promoted      : {report['trainable_reward_promoted']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$ValidatorPath = ".\05_training\rewards\validate_reward_pipeline_project_log_update_step130.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 130 self-test
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    updater = project_root / "05_training" / "rewards" / "update_project_log_reward_pipeline_step130.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_pipeline_project_log_update_step130.py"
    status = project_root / "05_training" / "rewards" / "reward_pipeline_project_log_update_step130.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_pipeline_project_log_update_step130_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_pipeline_project_log_update_step130_validation_{suffix}"

    update_cmd = [
        sys.executable,
        str(updater),
        "--status",
        str(status),
        "--project-log",
        str(project_root / "project_log.md"),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(update_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 130 project log updater failed")

    manifest_path = output_root / "reward_pipeline_project_log_update_step130_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 130 manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(validation_root),
    ]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 130 validator failed")

    report = load_json(validation_root / "reward_pipeline_project_log_update_step130_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_PROJECT_LOG_UPDATED_REWARD_PIPELINE_WAITING",
        "next_status": "READY_FOR_STEP131_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT_OR_PUSH_SYNC",
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    print("[OK] Step 130 reward pipeline project log update self-test PASS")
    print("[DONE] Step 130 reward pipeline project log update complete.")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\rewards\test_validate_reward_pipeline_project_log_update_step130.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 130 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 130 reward pipeline project log update self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 130 reward pipeline project log update files created and tested."
Write-Host "[NEXT] Review git status, then commit project_log.md and Step 130 files."
