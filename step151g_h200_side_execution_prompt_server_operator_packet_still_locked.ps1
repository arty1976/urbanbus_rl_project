$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) { throw "[STOP] Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot

$RewardDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardDir | Out-Null

$MdPath = "$RewardDir\h200_side_execution_prompt_server_operator_packet_step151g.md"
$PyPath = "$RewardDir\h200_side_execution_prompt_server_operator_packet_step151g.py"
$ValPath = "$RewardDir\validate_h200_side_execution_prompt_server_operator_packet_step151g.py"
$TestPath = "$RewardDir\test_h200_side_execution_prompt_server_operator_packet_step151g.py"

@'
# Step 151-G - H200-side execution prompt / server operator packet, still locked

This step creates a server-operator packet for the real H200-side Step 149 rerun.
It does not run H200 preflight, does not record operator approval, and does not release actual reward ablation execution.

Required H200 command:

```bash
python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \
  --project-root /workspace/urbanbus_rl_project \
  --expect-h200 \
  --min-gpu-count 1 \
  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual
```
'@ | Set-Content -Path $MdPath -Encoding UTF8

@'
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPECTED_STEP151E_STATUS = "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED"
EXPECTED_STEP151E_DECISION = "INDEX_ONLY_HANDOFF_NOT_RELEASED"
STEP151G_STATUS = "H200_SIDE_EXECUTION_PROMPT_PACKET_READY_STILL_LOCKED"
STEP151G_DECISION = "SERVER_OPERATOR_PACKET_ONLY_NOT_RELEASED"
DEFAULT_STEP151E_MANIFEST = "artifacts/rewards/h200_handoff_packet_index_step151e/h200_handoff_packet_index_step151e_manifest.json"
H200_COMMAND = "python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py --project-root /workspace/urbanbus_rl_project --expect-h200 --min-gpu-count 1 --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual"
H200_EXPECTED_RESULT_MANIFEST = "artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual/h200_environment_preflight_result_manifest_step149.json"


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def run_git(project_root: Path, args: List[str]) -> str:
    try:
        cp = subprocess.run(["git", *args], cwd=str(project_root), text=True, capture_output=True, check=False)
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def add_check(checks: List[Dict[str, Any]], check_id: str, passed: bool, severity: str, message: str, evidence: Optional[Dict[str, Any]] = None) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "severity": severity, "message": message, "evidence": evidence or {}})


def build_operator_prompt() -> str:
    return f"""# H200 server operator packet - Step 151-G

You are operating on the real H200 server.

## Goal
Run only the H200 environment preflight rerun. Do not start reward ablation training.

## Required working directory
```bash
cd /workspace/urbanbus_rl_project
```

## Record git state
```bash
git rev-parse HEAD
git status --short
```

## Run Step 149 in real H200 expected mode
```bash
{H200_COMMAND}
```

## Expected output manifest
```text
{H200_EXPECTED_RESULT_MANIFEST}
```

## Required result
- audit_status = PASS
- expect_h200 = true
- torch_import_ok = true
- cuda_available = true
- gpu_count >= 1
- hard_failures = 0
- actual_execution_allowed = false
- train_allowed = false

## Forbidden actions
Do not run actual 72-run reward ablation.
Do not set actual_execution_allowed=true.
Do not set actual_execution_released=true.
Do not set train_allowed=true.
Do not select a winner.
Do not promote a trainable reward.
Do not make paper-level or causal performance claims.
"""


def build_manifest(project_root: Path, step151e_manifest_path: Path, output_root: Path) -> Dict[str, Any]:
    if not step151e_manifest_path.exists():
        raise FileNotFoundError(f"Step 151-E manifest not found: {step151e_manifest_path}")

    step151e = load_json(step151e_manifest_path)
    checks: List[Dict[str, Any]] = []
    status = str(step151e.get("handoff_index_status", ""))
    decision = str(step151e.get("decision", ""))
    hard_failures = coerce_int(step151e.get("hard_failures", 999), default=999)

    add_check(checks, "step151e_status_ready", status == EXPECTED_STEP151E_STATUS, "hard", f"Step 151-E handoff_index_status must be {EXPECTED_STEP151E_STATUS}; got {status}")
    add_check(checks, "step151e_decision_index_only", decision == EXPECTED_STEP151E_DECISION, "hard", f"Step 151-E decision must be {EXPECTED_STEP151E_DECISION}; got {decision}")
    add_check(checks, "step151e_hard_failures_zero", hard_failures == 0, "hard", f"Step 151-E hard_failures must be 0; got {hard_failures}")

    lock_fields = ["operator_approval_recorded", "operator_approval_granted", "actual_execution_allowed", "actual_execution_released", "train_allowed", "actual_results", "winner_selected", "trainable_reward_promoted", "paper_level_claim_allowed", "causal_performance_claim_allowed"]
    lock_values: Dict[str, Any] = {}
    lock_ok = True
    for field in lock_fields:
        value = coerce_bool(step151e.get(field), default=None)
        lock_values[field] = value
        if value is not False:
            lock_ok = False
    add_check(checks, "step151e_locks_false", lock_ok, "hard", "Step 151-E must keep all approval/execution/result/claim locks false", lock_values)
    add_check(checks, "h200_command_includes_expected_flags", "--expect-h200" in H200_COMMAND and "--min-gpu-count 1" in H200_COMMAND, "hard", "H200 command must include --expect-h200 and --min-gpu-count 1", {"command": H200_COMMAND})

    git_status_short = run_git(project_root, ["status", "--short"])
    add_check(checks, "git_status_recorded", git_status_short != "UNKNOWN", "warning", "Git worktree status is recorded for operator packet review", {"git_status_short": git_status_short})

    hard_failure_count = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])
    packet_status = STEP151G_STATUS if hard_failure_count == 0 else "BLOCKED"

    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = output_root / "h200_server_operator_prompt_step151g.md"
    command_path = output_root / "h200_step149_expected_preflight_command_step151g.sh"
    manifest_path = output_root / "h200_side_execution_prompt_server_operator_packet_step151g_manifest.json"
    summary_path = output_root / "h200_side_execution_prompt_server_operator_packet_step151g_summary.md"

    write_text(prompt_path, build_operator_prompt())
    write_text(command_path, "#!/usr/bin/env bash\nset -euo pipefail\ncd /workspace/urbanbus_rl_project\n" + H200_COMMAND + "\n")

    manifest: Dict[str, Any] = {
        "artifact_version": "h200_side_execution_prompt_server_operator_packet_step151g_v3",
        "step": "151-G",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "git_commit": run_git(project_root, ["rev-parse", "HEAD"]),
        "git_status_short": git_status_short,
        "input_step": "151-E",
        "step151e_manifest": str(step151e_manifest_path),
        "packet_status": packet_status,
        "decision": STEP151G_DECISION,
        "hard_failures": int(hard_failure_count),
        "warnings": int(warning_count),
        "checks": checks,
        "h200_command": H200_COMMAND,
        "h200_expected_result_manifest": H200_EXPECTED_RESULT_MANIFEST,
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
        "server_operator_packet_only": True,
        "actual_release_requires_separate_manifest": True,
        "next_gate": "RUN_REAL_H200_STEP149_EXPECT_H200_OR_SYNC_RESULT_BACK_TO_STEP151C",
        "output_files": {"manifest": str(manifest_path), "summary": str(summary_path), "operator_prompt": str(prompt_path), "h200_command_script": str(command_path)},
    }

    dump_json(manifest_path, manifest)
    write_text(summary_path, "\n".join([
        "# Step 151-G H200-side server operator packet summary", "",
        f"- packet_status: {packet_status}", f"- decision: {STEP151G_DECISION}",
        f"- hard_failures: {hard_failure_count}", f"- warnings: {warning_count}",
        "- actual_execution_allowed: False", "- actual_execution_released: False", "- train_allowed: False", "",
        "## H200 command", "", "```bash", H200_COMMAND, "```", "",
        "## Output files", "", f"- operator_prompt: `{prompt_path}`", f"- h200_command_script: `{command_path}`", f"- manifest: `{manifest_path}`", "",
    ]))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--step151e-manifest", default=DEFAULT_STEP151E_MANIFEST)
    parser.add_argument("--output-root", default="artifacts/rewards/h200_side_execution_prompt_server_operator_packet_step151g")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    step151e_manifest_path = Path(args.step151e_manifest)
    if not step151e_manifest_path.is_absolute():
        step151e_manifest_path = project_root / step151e_manifest_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    manifest = build_manifest(project_root, step151e_manifest_path, output_root)
    print("[OK] Step 151-G H200-side execution prompt / server operator packet completed")
    print(f"[OK] packet_status    : {manifest['packet_status']}")
    print(f"[OK] decision         : {manifest['decision']}")
    print(f"[OK] hard_failures    : {manifest['hard_failures']}")
    print(f"[OK] warnings         : {manifest['warnings']}")
    print(f"[OK] operator_approval_recorded : {manifest['operator_approval_recorded']}")
    print(f"[OK] actual_execution_allowed   : {manifest['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released  : {manifest['actual_execution_released']}")
    print(f"[OK] train_allowed              : {manifest['train_allowed']}")
    print(f"[OK] operator_prompt : {manifest['output_files']['operator_prompt']}")
    print(f"[OK] command_script  : {manifest['output_files']['h200_command_script']}")
    print(f"[OK] manifest        : {manifest['output_files']['manifest']}")
    if manifest["hard_failures"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@ | Set-Content -Path $PyPath -Encoding UTF8

@'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "H200_SIDE_EXECUTION_PROMPT_PACKET_READY_STILL_LOCKED"
EXPECTED_DECISION = "SERVER_OPERATOR_PACKET_ONLY_NOT_RELEASED"


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def validate(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    require(manifest.get("step") == "151-G", "step must be 151-G", errors)
    require(manifest.get("packet_status") == EXPECTED_STATUS, f"packet_status must be {EXPECTED_STATUS}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)
    for field in ["operator_approval_recorded", "operator_approval_granted", "actual_execution_allowed", "actual_execution_released", "train_allowed", "actual_results", "winner_selected", "trainable_reward_promoted", "paper_level_claim_allowed", "causal_performance_claim_allowed"]:
        require(manifest.get(field) is False, f"{field} must be false", errors)
    require(manifest.get("server_operator_packet_only") is True, "server_operator_packet_only must be true", errors)
    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)
    command = str(manifest.get("h200_command", ""))
    require("--expect-h200" in command, "h200_command must include --expect-h200", errors)
    require("--min-gpu-count 1" in command, "h200_command must include --min-gpu-count 1", errors)
    output_files = manifest.get("output_files", {})
    for key in ("manifest", "summary", "operator_prompt", "h200_command_script"):
        require(bool(output_files.get(key)), f"output_files.{key} must be recorded", errors)
    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 5, "checks must contain at least 5 entries", errors)
    for check in checks:
        if check.get("severity") == "hard":
            require(check.get("passed") is True, f"hard check failed: {check.get('check_id')}", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    path = Path(args.manifest)
    errors = validate(load_json(path))
    if errors:
        print("[FAIL] Step 151-G manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)
    print("[OK] Step 151-G manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
'@ | Set-Content -Path $ValPath -Encoding UTF8

@'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {cmd}")
    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed: {cmd}")
    return cp


def good_151e() -> dict:
    return {
        "artifact_version": "h200_handoff_packet_index_step151e_v1",
        "step": "151-E",
        "handoff_index_status": "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED",
        "decision": "INDEX_ONLY_HANDOFF_NOT_RELEASED",
        "hard_failures": 0,
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "h200_side_execution_prompt_server_operator_packet_step151g.py"
    val = reward_dir / "validate_h200_side_execution_prompt_server_operator_packet_step151g.py"
    selftest_root = project_root / "artifacts" / "rewards" / "h200_side_execution_prompt_server_operator_packet_step151g_selftest"
    input_root = selftest_root / "input"
    good = input_root / "step151e_good.json"
    bad = input_root / "step151e_bad.json"
    dump_json(good, good_151e())
    bad_payload = good_151e()
    bad_payload["actual_execution_allowed"] = True
    dump_json(bad, bad_payload)
    positive_output = selftest_root / "positive_output"
    run([sys.executable, str(gen), "--project-root", str(project_root), "--step151e-manifest", str(good), "--output-root", str(positive_output)], cwd=project_root, expect_ok=True)
    manifest = positive_output / "h200_side_execution_prompt_server_operator_packet_step151g_manifest.json"
    run([sys.executable, str(val), "--manifest", str(manifest)], cwd=project_root, expect_ok=True)
    run([sys.executable, str(gen), "--project-root", str(project_root), "--step151e-manifest", str(bad), "--output-root", str(selftest_root / "negative_output")], cwd=project_root, expect_ok=False)
    print("[OK] Step 151-G H200-side execution prompt server operator packet self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
'@ | Set-Content -Path $TestPath -Encoding UTF8

foreach ($required in @($MdPath, $PyPath, $ValPath, $TestPath)) {
    if (-not (Test-Path $required)) { throw "[STOP] failed to create Step 151-G file: $required" }
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 151-G self-test failed" }

$Step151EManifest = ".\artifacts\rewards\h200_handoff_packet_index_step151e\h200_handoff_packet_index_step151e_manifest.json"
if (-not (Test-Path $Step151EManifest)) { throw "[STOP] Step 151-E manifest not found: $Step151EManifest" }

& $py $PyPath `
  --project-root "." `
  --step151e-manifest $Step151EManifest `
  --output-root ".\artifacts\rewards\h200_side_execution_prompt_server_operator_packet_step151g"

if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 151-G packet generation failed" }

$Step151GManifest = ".\artifacts\rewards\h200_side_execution_prompt_server_operator_packet_step151g\h200_side_execution_prompt_server_operator_packet_step151g_manifest.json"

& $py $ValPath --manifest $Step151GManifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 151-G manifest validation failed" }

Write-Host "[DONE] Step 151-G H200-side execution prompt server operator packet complete."
Write-Host "[OK] manifest: $Step151GManifest"
