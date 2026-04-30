$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = Join-Path $ProjectRoot "05_training\rewards"
if (-not (Test-Path $RewardsDir)) {
    New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null
}

$PyMain = Join-Path $RewardsDir "run_actual_reward_ablation_candidate.py"
$PyValidate = Join-Path $RewardsDir "validate_actual_reward_ablation_runner_guard_step133.py"
$PyTest = Join-Path $RewardsDir "test_actual_reward_ablation_runner_guard_step133.py"

$MainCode = @'
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STEP_ID = 133
ARTIFACT_VERSION = "actual_reward_ablation_runner_guard_step133_v1"

VALID_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
VALID_CONDITIONS = ["A"]

FORBIDDEN_TRUE_GUARDS = [
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def validate_step132_manifest(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "provided": bool(path),
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "status": "not_provided",
        "warnings": [],
        "violations": [],
    }

    if path is None:
        result["warnings"].append("step132_manifest_not_provided")
        return result

    if not path.exists():
        result["status"] = "missing"
        result["warnings"].append("step132_manifest_missing")
        return result

    try:
        payload = load_json_any(path)
    except Exception as exc:
        result["status"] = "unreadable"
        result["violations"].append(f"step132_manifest_unreadable: {exc}")
        return result

    result["status"] = "read"
    result["audit_status"] = payload.get("audit_status")
    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")

    # Step 132 is expected to be a dry-run only layer. If any actual-result
    # signal is true, Step 133 must refuse to treat it as a safe upstream state.
    for key in FORBIDDEN_TRUE_GUARDS:
        if bool(payload.get(key, False)):
            result["violations"].append(f"forbidden_true_in_step132_manifest: {key}")

    decision = payload.get("decision", {})
    if isinstance(decision, dict):
        for key in FORBIDDEN_TRUE_GUARDS:
            if bool(decision.get(key, False)):
                result["violations"].append(f"forbidden_true_in_step132_decision: {key}")

    non_claim_guards = payload.get("non_claim_guards", {})
    if isinstance(non_claim_guards, dict):
        for key in FORBIDDEN_TRUE_GUARDS:
            if bool(non_claim_guards.get(key, False)):
                result["violations"].append(f"forbidden_true_in_step132_non_claim_guards: {key}")

    return result


def make_non_claim_guards() -> Dict[str, bool]:
    return {key: False for key in FORBIDDEN_TRUE_GUARDS}


def build_runner_guard_manifest(
    *,
    project_root: Path,
    reward_id: str,
    seed: int,
    condition: str,
    mode: str,
    output_root: Path,
    execute_actual: bool,
    step132_manifest: Optional[Path],
    candidate_label: str = "",
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    reward_id = str(reward_id).strip().upper()
    condition = str(condition).strip().upper()
    mode = str(mode).strip().lower()

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    if reward_id not in VALID_REWARD_IDS:
        blocking_reasons.append(f"invalid_reward_id: {reward_id}")

    if condition not in VALID_CONDITIONS:
        blocking_reasons.append(f"invalid_condition_for_reward_ablation_guard: {condition}")

    if int(seed) <= 0:
        blocking_reasons.append("seed_must_be_positive")

    if mode not in ("dry-run", "guard", "actual"):
        blocking_reasons.append(f"invalid_mode: {mode}")

    # Step 133 deliberately blocks actual execution. It only proves that the
    # runner entrypoint exists and that unsafe actual execution requests are
    # stopped before writing result tables.
    if mode == "actual" or execute_actual:
        blocking_reasons.append("actual_execution_blocked_by_step133_guard")

    upstream = validate_step132_manifest(step132_manifest)
    warnings.extend(upstream.get("warnings", []))
    blocking_reasons.extend(upstream.get("violations", []))

    status = "PASS" if not blocking_reasons else "BLOCKED"

    candidate_key = f"{condition}_{reward_id}_seed_{int(seed):03d}"
    if candidate_label:
        candidate_key = f"{candidate_key}_{candidate_label}"

    manifest_path = output_root / "actual_reward_ablation_runner_guard_step133_manifest.json"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "runner_entrypoint": safe_rel(Path(__file__), project_root),
        "candidate": {
            "condition": condition,
            "reward_id": reward_id,
            "seed": int(seed),
            "candidate_key": candidate_key,
        },
        "execution_request": {
            "mode": mode,
            "execute_actual": bool(execute_actual),
        },
        "audit_status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "upstream_step132_check": upstream,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "non_claim_guards": make_non_claim_guards(),
        "scope_note": (
            "Step 133 creates a guarded runner entrypoint for reward ablation candidates. "
            "It does not run actual ablation, does not write actual result rows, does not select a winner, "
            "and does not promote any trainable reward."
        ),
    }

    dump_json(manifest_path, payload)

    latest_path = project_root / "05_training" / "rewards" / "actual_reward_ablation_runner_guard_step133.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "audit_status": status,
        "candidate_key": candidate_key,
        "actual_executed": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--reward-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="A")
    parser.add_argument("--mode", default="dry-run", choices=["dry-run", "guard", "actual"])
    parser.add_argument("--execute-actual", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step132-manifest", default="")
    parser.add_argument("--candidate-label", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    step132_manifest = Path(args.step132_manifest) if args.step132_manifest else None
    if step132_manifest is not None and not step132_manifest.is_absolute():
        step132_manifest = project_root / step132_manifest

    payload = build_runner_guard_manifest(
        project_root=project_root,
        reward_id=args.reward_id,
        seed=args.seed,
        condition=args.condition,
        mode=args.mode,
        output_root=output_root,
        execute_actual=args.execute_actual,
        step132_manifest=step132_manifest,
        candidate_label=args.candidate_label,
    )

    print("[OK] Step 133 actual reward ablation runner guard completed")
    print(f"[OK] audit_status      : {payload['audit_status']}")
    print(f"[OK] candidate_key     : {payload['candidate']['candidate_key']}")
    print(f"[OK] actual_executed   : {payload['actual_executed']}")
    print(f"[OK] actual_results    : {payload['actual_results']}")
    print(f"[OK] winner_selected   : {payload['winner_selected']}")
    print(f"[OK] training_allowed  : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim       : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest          : {payload['manifest_path']}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 133 runner guard blocked this request")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@

$ValidateCode = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "actual_reward_ablation_runner_guard_step133_v1"
FORBIDDEN_TRUE_KEYS = [
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")

    if payload.get("step") != 133:
        errors.append("step_must_be_133")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    candidate = payload.get("candidate", {})
    if not isinstance(candidate, dict):
        errors.append("candidate_missing_or_invalid")
    else:
        if candidate.get("reward_id") not in ("R0", "R1", "R2", "R3", "R4", "R5"):
            errors.append("invalid_reward_id")
        if candidate.get("condition") != "A":
            errors.append("condition_must_be_A")
        if int(candidate.get("seed", 0)) <= 0:
            errors.append("seed_must_be_positive")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    guards = payload.get("non_claim_guards", {})
    if not isinstance(guards, dict):
        errors.append("non_claim_guards_missing_or_invalid")
    else:
        for key, value in guards.items():
            if bool(value):
                errors.append(f"forbidden_true_non_claim_guard: {key}")

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"[FAIL] {err}")
        raise SystemExit(2)

    print("[OK] Step 133 actual reward ablation runner guard manifest validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from run_actual_reward_ablation_candidate import build_runner_guard_manifest
from validate_actual_reward_ablation_runner_guard_step133 import validate_payload


def test_dry_run_pass() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_pass_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "PASS", payload
        assert payload["actual_executed"] is False
        assert payload["actual_results"] is False
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_actual_attempt_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_actual_block_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="actual",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "BLOCKED"
        assert "actual_execution_blocked_by_step133_guard" in payload["blocking_reasons"]
        result = validate_payload(payload, require_pass=False)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_reward_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_invalid_reward_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R9",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "BLOCKED"
        assert any("invalid_reward_id" in r for r in payload["blocking_reasons"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validator_catches_tampered_actual_results() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_tamper_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        payload["actual_results"] = True
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "FAIL"
        assert any("actual_results" in e for e in result["errors"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_dry_run_pass()
    test_actual_attempt_blocks()
    test_invalid_reward_blocks()
    test_validator_catches_tampered_actual_results()
    print("[OK] Step 133 actual reward ablation runner guard self-test PASS")


if __name__ == "__main__":
    main()
'@

Set-Content -Path $PyMain -Value $MainCode -Encoding UTF8
Set-Content -Path $PyValidate -Value $ValidateCode -Encoding UTF8
Set-Content -Path $PyTest -Value $TestCode -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = "artifacts/rewards/actual_reward_ablation_runner_guard_step133/sample_R0_seed001"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\actual_reward_ablation_runner_guard_step133_manifest.json"

& $py $PyMain `
  --project-root "." `
  --reward-id R0 `
  --seed 1 `
  --condition A `
  --mode dry-run `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 133 runner guard sample dry-run failed"
}

& $py $PyValidate --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 133 manifest validation failed"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 133 self-test failed"
}

Write-Host "[DONE] Step 133 actual reward ablation runner guard complete."
