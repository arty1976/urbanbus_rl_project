from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 135
ARTIFACT_VERSION = "reward_ablation_actual_execution_readiness_lock_step135_v1"

REQUIRED_POINTERS = [
    {
        "name": "step131_environment_preflight",
        "step": 131,
        "pointer": "05_training/rewards/reward_ablation_execution_environment_preflight_step131.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step131*manifest*.json",
    },
    {
        "name": "step132_command_dry_run",
        "step": 132,
        "pointer": "05_training/rewards/reward_ablation_command_dry_run_executor_step132.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step132*manifest*.json",
    },
    {
        "name": "step134_command_runner_integration",
        "step": 134,
        "pointer": "05_training/rewards/reward_ablation_command_runner_integration_step134.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step134*manifest*.json",
    },
]

FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
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


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = project_root / p
    return p


def find_manifest_from_pointer(project_root: Path, pointer_rel: str, fallback_glob: str) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / pointer_rel

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            raw = payload.get("manifest_path") or payload.get("path") or payload.get("latest_manifest_path")
            if raw:
                p = resolve_path(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append(f"pointer_manifest_missing: {pointer_rel}")
            else:
                warnings.append(f"pointer_has_no_manifest_path: {pointer_rel}")
        except Exception as exc:
            warnings.append(f"pointer_unreadable: {pointer_rel}: {exc}")
    else:
        warnings.append(f"pointer_missing: {pointer_rel}")

    candidates = sorted(
        [p for p in project_root.glob(fallback_glob) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        warnings.append(f"using_fallback_manifest_search: {fallback_glob}")
        return candidates[0], warnings

    return None, warnings


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))

    return violations


def check_required_manifest(project_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path, warnings = find_manifest_from_pointer(
        project_root,
        spec["pointer"],
        spec["fallback_glob"],
    )

    result: Dict[str, Any] = {
        "name": spec["name"],
        "required_step": spec["step"],
        "pointer": spec["pointer"],
        "manifest_path": str(manifest_path) if manifest_path else "",
        "exists": bool(manifest_path and manifest_path.exists()),
        "warnings": warnings,
        "violations": [],
        "summary": {},
    }

    if manifest_path is None or not manifest_path.exists():
        result["violations"].append(f"required_manifest_missing: {spec['name']}")
        return result

    try:
        payload = load_json_any(manifest_path)
    except Exception as exc:
        result["violations"].append(f"required_manifest_unreadable: {spec['name']}: {exc}")
        return result

    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")
    result["summary"] = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}

    if payload.get("step") != spec["step"]:
        result["violations"].append(
            f"manifest_step_mismatch: {spec['name']}: expected={spec['step']} got={payload.get('step')}"
        )

    if payload.get("audit_status") != "PASS":
        result["violations"].append(
            f"manifest_audit_not_pass: {spec['name']}: {payload.get('audit_status')}"
        )

    result["violations"].extend(recursive_forbidden_true(payload))

    # Specific readiness checks for the two command-level gates.
    if spec["step"] == 132:
        planned_count = None
        if isinstance(payload.get("summary"), dict):
            planned_count = payload["summary"].get("planned_command_count")
        if planned_count is None:
            planned_count = payload.get("planned_command_count") or payload.get("planned_commands_count")
        if planned_count is not None and int(planned_count) != 18:
            result["violations"].append(f"step132_planned_command_count_not_18: {planned_count}")

    if spec["step"] == 134:
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            result["violations"].append("step134_summary_missing")
        else:
            expected = {
                "planned_command_count": 18,
                "step133_invocation_count": 18,
                "step133_guard_pass_count": 18,
                "step133_guard_fail_count": 0,
            }
            for key, expected_value in expected.items():
                actual_value = int(summary.get(key, -999))
                if actual_value != expected_value:
                    result["violations"].append(
                        f"step134_summary_mismatch: {key}: expected={expected_value} got={actual_value}"
                    )

    return result


def check_runner_files(project_root: Path) -> Dict[str, Any]:
    expected = [
        "05_training/rewards/run_actual_reward_ablation_candidate.py",
        "05_training/rewards/reward_ablation_command_runner_integration_step134.py",
    ]

    missing = []
    present = []
    for rel in expected:
        p = project_root / rel
        if p.exists():
            present.append(rel)
        else:
            missing.append(rel)

    return {
        "expected_files": expected,
        "present_files": present,
        "missing_files": missing,
        "violations": [f"required_runner_file_missing: {x}" for x in missing],
    }


def build_release_requirements() -> List[Dict[str, Any]]:
    return [
        {
            "requirement_id": "REQ-001",
            "name": "operator_explicit_release",
            "description": "A future Step 136 release manifest must explicitly authorize actual execution.",
            "satisfied": False,
        },
        {
            "requirement_id": "REQ-002",
            "name": "actual_runner_mode_review",
            "description": "The guarded runner must be reviewed before allowing any --mode actual path.",
            "satisfied": False,
        },
        {
            "requirement_id": "REQ-003",
            "name": "output_storage_review",
            "description": "Actual result output storage and retention path must be confirmed.",
            "satisfied": False,
        },
        {
            "requirement_id": "REQ-004",
            "name": "non_claim_guard_review",
            "description": "Winner selection, promotion, training and paper-claim flags must remain false until actual results are ingested.",
            "satisfied": False,
        },
        {
            "requirement_id": "REQ-005",
            "name": "post_execution_ingestion_plan",
            "description": "A future ingestion step must validate actual outputs before any winner or promotion decision.",
            "satisfied": False,
        },
    ]


def run_readiness_lock(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    upstream_checks = [check_required_manifest(project_root, spec) for spec in REQUIRED_POINTERS]
    runner_check = check_runner_files(project_root)

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for check in upstream_checks:
        blocking_reasons.extend(check.get("violations", []))
        warnings.extend(check.get("warnings", []))

    blocking_reasons.extend(runner_check.get("violations", []))

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"

    readiness_lock_status = (
        "LOCKED_READY_FOR_EXPLICIT_RELEASE"
        if audit_status == "PASS"
        else "LOCKED_BLOCKED"
    )

    release_requirements = build_release_requirements()

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "readiness_lock_status": readiness_lock_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "upstream_checks": upstream_checks,
        "runner_file_check": runner_check,
        "release_requirements": release_requirements,
        "actual_execution_allowed": False,
        "actual_execution_release_step_required": 136,
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
        "scope_note": (
            "Step 135 locks the reward ablation execution readiness state. "
            "Passing this step means the dry-run command path and guarded runner path are ready, "
            "but actual execution remains disabled until a future explicit release step."
        ),
    }

    manifest_path = output_root / "reward_ablation_actual_execution_readiness_lock_step135_manifest.json"
    dump_json(manifest_path, payload)

    latest_path = project_root / "05_training" / "rewards" / "reward_ablation_actual_execution_readiness_lock_step135.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "readiness_lock_status": readiness_lock_status,
        "actual_execution_allowed": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_execution_readiness_lock_step135")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_readiness_lock(project_root, output_root)

    print("[OK] Step 135 reward ablation actual execution readiness lock completed")
    print(f"[OK] audit_status             : {payload['audit_status']}")
    print(f"[OK] readiness_lock_status    : {payload['readiness_lock_status']}")
    print(f"[OK] actual_execution_allowed : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_executed          : {payload['actual_executed']}")
    print(f"[OK] actual_results           : {payload['actual_results']}")
    print(f"[OK] winner_selected          : {payload['winner_selected']}")
    print(f"[OK] training_allowed         : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim              : {payload['paper_level_claim_allowed']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 135 readiness lock failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
