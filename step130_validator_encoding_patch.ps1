$ErrorActionPreference = "Stop"

Set-Location "C:\Users\ryujo\urbanbus_rl_project"

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


def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    expected: Any,
    actual: Any,
    description: str,
) -> None:
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
    add_check(
        checks,
        "log_update_status",
        "PROJECT_LOG_UPDATED",
        manifest.get("log_update_status"),
        "Project log must be updated.",
    )
    add_check(
        checks,
        "project_log_exists",
        True,
        project_log.exists(),
        "project_log.md must exist.",
    )

    text = project_log.read_text(encoding="utf-8-sig") if project_log.exists() else ""

    add_check(
        checks,
        "section_marker_present",
        True,
        SECTION_MARKER in text,
        "Step 130 marker must exist.",
    )
    add_check(
        checks,
        "section_end_present",
        True,
        SECTION_END in text,
        "Step 130 end marker must exist.",
    )

    for step in EXPECTED_STEPS:
        found = f"Step{step}" in text or f"Step {step}" in text
        add_check(
            checks,
            f"log_mentions_step_{step}",
            True,
            found,
            f"project_log must mention Step {step}.",
        )

    # ASCII-only phrase checks to avoid PowerShell/Korean encoding corruption.
    required_phrases = [
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "R0",
        "R5",
        "72",
    ]
    for phrase in required_phrases:
        add_check(
            checks,
            f"phrase_{phrase}",
            True,
            phrase in text,
            f"project_log must include phrase: {phrase}",
        )

    for key in [
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        add_check(
            checks,
            f"manifest_guard_{key}",
            False,
            bool(manifest.get(key)),
            f"{key} must remain false.",
        )

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
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_pipeline_project_log_update_step130_validation",
    )
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

Write-Host "[OK] Step 130 validator rewritten with ASCII-safe phrase checks"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$TestPath = ".\05_training\rewards\test_validate_reward_pipeline_project_log_update_step130.py"

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 130 self-test failed after validator encoding patch"
}

Write-Host "[DONE] Step 130 validator encoding patch and self-test PASS"
