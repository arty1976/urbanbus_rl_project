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

$PyMain = Join-Path $RewardsDir "reward_ablation_command_dry_run_executor_step132.py"
$PyValidate = Join-Path $RewardsDir "validate_reward_ablation_command_dry_run_executor_step132.py"
$PyTest = Join-Path $RewardsDir "test_validate_reward_ablation_command_dry_run_executor_step132.py"

$MainCode = @'
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STEP_ID = 132
ARTIFACT_VERSION = "reward_ablation_command_dry_run_executor_step132_v1"
STEP131_ARTIFACT_VERSION = "reward_ablation_execution_environment_preflight_step131_v1"

DEFAULT_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_CONDITIONS = ["A"]

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


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv(value: str, default: List[str]) -> List[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def split_int_csv(value: str, default: List[int]) -> List[int]:
    if value is None or str(value).strip() == "":
        return list(default)
    out: List[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw))
    return out


def find_step131_manifest(project_root: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = project_root / p
        return p

    pointer = project_root / "05_training" / "rewards" / "reward_ablation_execution_environment_preflight_step131.latest.json"
    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            manifest_path = Path(str(payload.get("manifest_path", "")))
            if not manifest_path.is_absolute():
                manifest_path = project_root / manifest_path
            return manifest_path
        except Exception:
            pass

    fallback = (
        project_root
        / "artifacts"
        / "rewards"
        / "reward_ablation_execution_environment_preflight_step131"
        / "reward_ablation_execution_environment_preflight_step131_manifest.json"
    )
    if fallback.exists():
        return fallback

    return None


def validate_step131_manifest(project_root: Path, explicit_path: Optional[str]) -> Dict[str, Any]:
    manifest_path = find_step131_manifest(project_root, explicit_path)
    result: Dict[str, Any] = {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "exists": bool(manifest_path and manifest_path.exists()),
        "status": "UNKNOWN",
        "errors": [],
        "payload_summary": {},
    }

    if manifest_path is None or not manifest_path.exists():
        result["status"] = "BLOCKED"
        result["errors"].append("step131_manifest_missing")
        return result

    try:
        payload = load_json_any(manifest_path)
    except Exception as exc:
        result["status"] = "BLOCKED"
        result["errors"].append(f"step131_manifest_unreadable: {exc}")
        return result

    result["payload_summary"] = {
        "artifact_version": payload.get("artifact_version"),
        "step": payload.get("step"),
        "audit_status": payload.get("audit_status"),
        "actual_reward_ablation_execution_allowed": payload.get("decision", {}).get(
            "actual_reward_ablation_execution_allowed"
        ),
    }

    if payload.get("artifact_version") != STEP131_ARTIFACT_VERSION:
        result["errors"].append("step131_artifact_version_mismatch")
    if payload.get("step") != 131:
        result["errors"].append("step131_step_mismatch")
    if payload.get("audit_status") != "PASS":
        result["errors"].append("step131_audit_status_not_pass")
    if payload.get("decision", {}).get("actual_reward_ablation_execution_allowed") is not True:
        result["errors"].append("step131_execution_not_allowed")

    guards = payload.get("non_claim_guards", {})
    for key in FORBIDDEN_TRUE_GUARDS:
        if bool(guards.get(key, False)):
            result["errors"].append(f"step131_forbidden_true_non_claim_guard: {key}")

    result["status"] = "PASS" if not result["errors"] else "BLOCKED"
    return result


def build_command(
    project_root: Path,
    candidate_id: str,
    condition_id: str,
    seed: int,
    output_root: Path,
    runner_path: str,
) -> Dict[str, Any]:
    run_id = f"{candidate_id}_{condition_id}_seed_{seed:03d}"
    run_output = output_root / "planned_runs" / run_id

    runner = Path(runner_path)
    if not runner.is_absolute():
        runner = project_root / runner

    cmd_parts = [
        sys.executable,
        str(runner),
        "--candidate-id",
        candidate_id,
        "--condition-id",
        condition_id,
        "--seed",
        str(seed),
        "--output-root",
        str(run_output),
        "--actual-result-schema",
        "step124_bridge",
        "--selection-gate",
        "step125_selection_criteria",
        "--dry-run-only",
    ]

    return {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "condition_id": condition_id,
        "seed": int(seed),
        "runner_path": str(runner),
        "runner_exists": runner.exists(),
        "output_root": str(run_output),
        "command": " ".join(shlex.quote(x) for x in cmd_parts),
        "execution_status": "NOT_EXECUTED_DRY_RUN_ONLY",
        "actual_result_written": False,
        "winner_selection_allowed": False,
        "training_allowed": False,
    }


def run_step132(
    project_root: Path,
    output_root: Path,
    step131_manifest: Optional[str],
    candidates: List[str],
    conditions: List[str],
    seeds: List[int],
    runner_path: str,
    require_runner_exists: bool,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    step131 = validate_step131_manifest(project_root, step131_manifest)
    commands: List[Dict[str, Any]] = []
    blocking_reasons: List[str] = []
    warnings: List[str] = []

    if step131["status"] != "PASS":
        blocking_reasons.extend(step131["errors"])

    for candidate_id in candidates:
        if not candidate_id.upper().startswith("R"):
            blocking_reasons.append(f"invalid_candidate_id: {candidate_id}")

    for condition_id in conditions:
        if condition_id.upper() != "A":
            warnings.append(
                f"condition_not_A: {condition_id}; Step132 command dry-run can plan it, but current reward ablation scope is expected to use A unless later expanded."
            )

    for seed in seeds:
        if int(seed) < 0:
            blocking_reasons.append(f"invalid_seed: {seed}")

    for candidate_id in candidates:
        for condition_id in conditions:
            for seed in seeds:
                commands.append(
                    build_command(
                        project_root=project_root,
                        candidate_id=candidate_id.upper(),
                        condition_id=condition_id.upper(),
                        seed=int(seed),
                        output_root=output_root,
                        runner_path=runner_path,
                    )
                )

    missing_runners = [c for c in commands if not c["runner_exists"]]
    if missing_runners:
        msg = f"runner_path_missing_for_{len(missing_runners)}_planned_commands"
        if require_runner_exists:
            blocking_reasons.append(msg)
        else:
            warnings.append(msg)

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "audit_status": audit_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step131_manifest_check": step131,
        "planned_scope": {
            "candidates": candidates,
            "conditions": conditions,
            "seeds": seeds,
            "runner_path": runner_path,
            "require_runner_exists": bool(require_runner_exists),
        },
        "planned_command_count": int(len(commands)),
        "planned_commands": commands,
        "decision": {
            "command_dry_run_ready": audit_status == "PASS",
            "actual_execution_performed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "actual_training_allowed": False,
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "final_reward_design_claim_allowed": False,
            "best_reward_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "scope_note": (
            "Step 132 builds dry-run command manifests only. It does not execute reward ablation, "
            "does not ingest actual results, does not select a winner, and does not promote a reward."
        ),
    }

    manifest_path = output_root / "reward_ablation_command_dry_run_executor_step132_manifest.json"
    commands_path = output_root / "reward_ablation_command_dry_run_executor_step132_commands.json"
    dump_json(manifest_path, payload)
    dump_json(commands_path, {"planned_commands": commands})

    pointer_path = project_root / "05_training" / "rewards" / "reward_ablation_command_dry_run_executor_step132.latest.json"
    dump_json(pointer_path, {
        "manifest_path": str(manifest_path),
        "commands_path": str(commands_path),
        "audit_status": audit_status,
        "planned_command_count": int(len(commands)),
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_command_dry_run_executor_step132")
    parser.add_argument("--step131-manifest", default="")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument(
        "--runner-path",
        default="05_training/rewards/run_actual_reward_ablation_candidate.py",
    )
    parser.add_argument("--require-runner-exists", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_step132(
        project_root=project_root,
        output_root=output_root,
        step131_manifest=args.step131_manifest or None,
        candidates=split_csv(args.candidates, DEFAULT_CANDIDATES),
        conditions=split_csv(args.conditions, DEFAULT_CONDITIONS),
        seeds=split_int_csv(args.seeds, DEFAULT_SEEDS),
        runner_path=args.runner_path,
        require_runner_exists=bool(args.require_runner_exists),
    )

    print("[OK] Step 132 reward ablation command dry-run executor completed")
    print(f"[OK] audit_status        : {payload['audit_status']}")
    print(f"[OK] command_dry_run_ready: {payload['decision']['command_dry_run_ready']}")
    print(f"[OK] planned_commands   : {payload['planned_command_count']}")
    print(f"[OK] output_root        : {output_root}")
    print("[OK] actual_executed    : False")
    print("[OK] actual_results     : False")
    print("[OK] winner_selected    : False")
    print("[OK] training_allowed   : False")

    for warning in payload.get("warnings", []):
        print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 132 command dry-run failed")
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


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("step") != 132:
        errors.append("step_must_be_132")

    if payload.get("artifact_version") != "reward_ablation_command_dry_run_executor_step132_v1":
        errors.append("artifact_version_mismatch")

    audit_status = payload.get("audit_status")
    if audit_status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and audit_status != "PASS":
        errors.append("audit_status_not_pass")

    if require_pass and payload.get("decision", {}).get("command_dry_run_ready") is not True:
        errors.append("command_dry_run_ready_false_on_pass")

    if payload.get("decision", {}).get("actual_execution_performed") is not False:
        errors.append("actual_execution_must_be_false")

    if payload.get("planned_command_count") != len(payload.get("planned_commands", [])):
        errors.append("planned_command_count_mismatch")

    if require_pass and int(payload.get("planned_command_count", 0)) <= 0:
        errors.append("planned_command_count_must_be_positive")

    for cmd in payload.get("planned_commands", []):
        if cmd.get("execution_status") != "NOT_EXECUTED_DRY_RUN_ONLY":
            errors.append(f"planned_command_not_dry_run_only: {cmd.get('run_id')}")
        if bool(cmd.get("actual_result_written", True)):
            errors.append(f"actual_result_written_true: {cmd.get('run_id')}")
        if bool(cmd.get("winner_selection_allowed", True)):
            errors.append(f"winner_selection_allowed_true: {cmd.get('run_id')}")
        if bool(cmd.get("training_allowed", True)):
            errors.append(f"training_allowed_true: {cmd.get('run_id')}")

    guards = payload.get("non_claim_guards", {})
    for key in FORBIDDEN_TRUE_GUARDS:
        if bool(guards.get(key, False)):
            errors.append(f"forbidden_true_non_claim_guard: {key}")

    return {"validation_status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 132 reward ablation command dry-run manifest validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from reward_ablation_command_dry_run_executor_step132 import (
    DEFAULT_CANDIDATES,
    DEFAULT_CONDITIONS,
    DEFAULT_SEEDS,
    FORBIDDEN_TRUE_GUARDS,
    run_step132,
)
from validate_reward_ablation_command_dry_run_executor_step132 import validate_payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_step131_fixture(root: Path, pass_status: bool = True) -> Path:
    manifest = (
        root
        / "artifacts"
        / "rewards"
        / "reward_ablation_execution_environment_preflight_step131"
        / "reward_ablation_execution_environment_preflight_step131_manifest.json"
    )
    guards = {key: False for key in FORBIDDEN_TRUE_GUARDS}
    payload = {
        "artifact_version": "reward_ablation_execution_environment_preflight_step131_v1",
        "step": 131,
        "audit_status": "PASS" if pass_status else "BLOCKED",
        "decision": {
            "actual_reward_ablation_execution_allowed": bool(pass_status),
        },
        "non_claim_guards": guards,
    }
    write_json(manifest, payload)
    pointer = root / "05_training" / "rewards" / "reward_ablation_execution_environment_preflight_step131.latest.json"
    write_json(pointer, {"manifest_path": str(manifest), "audit_status": payload["audit_status"]})
    return manifest


def test_pass_fixture() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_pass_"))
    try:
        build_step131_fixture(tmp, pass_status=True)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0", "R1"],
            conditions=["A"],
            seeds=[1, 2],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=False,
        )
        assert payload["audit_status"] == "PASS", payload
        assert payload["planned_command_count"] == 4, payload
        assert payload["decision"]["actual_execution_performed"] is False
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_step131_blocked_blocks_step132() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_blocked_"))
    try:
        build_step131_fixture(tmp, pass_status=False)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0"],
            conditions=["A"],
            seeds=[1],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=False,
        )
        assert payload["audit_status"] == "BLOCKED", payload
        assert any("step131" in x for x in payload["blocking_reasons"]), payload
        result = validate_payload(payload, require_pass=False)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_required_blocks_when_missing() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_runner_missing_"))
    try:
        build_step131_fixture(tmp, pass_status=True)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0"],
            conditions=["A"],
            seeds=[1],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=True,
        )
        assert payload["audit_status"] == "BLOCKED", payload
        assert any("runner_path_missing" in x for x in payload["blocking_reasons"]), payload
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_pass_fixture()
    test_step131_blocked_blocks_step132()
    test_runner_required_blocks_when_missing()
    print("[OK] Step 132 reward ablation command dry-run executor self-test PASS")


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

$OutputRoot = ".\artifacts\rewards\reward_ablation_command_dry_run_executor_step132"
$Manifest = Join-Path $OutputRoot "reward_ablation_command_dry_run_executor_step132_manifest.json"

& $py $PyMain --project-root "." --output-root "artifacts/rewards/reward_ablation_command_dry_run_executor_step132"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 132 command dry-run executor failed"
}

& $py $PyValidate --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 132 manifest validation failed"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 132 self-test failed"
}

Write-Host "[DONE] Step 132 reward ablation command dry-run executor complete."
