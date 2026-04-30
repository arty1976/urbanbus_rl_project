from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 136
ARTIFACT_VERSION = "reward_ablation_actual_execution_release_request_step136_v1"

STEP135_POINTER = "05_training/rewards/reward_ablation_actual_execution_readiness_lock_step135.latest.json"
STEP135_FALLBACK_GLOB = "artifacts/rewards/**/*step135*manifest*.json"

EXPECTED_MATRIX = {
    "conditions": ["A"],
    "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
    "seeds": [1, 2, 3],
    "planned_run_count": 18,
}

FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
    "actual_execution_released",
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


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def find_step135_manifest(project_root: Path) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP135_POINTER

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            raw = payload.get("manifest_path") or payload.get("path") or payload.get("latest_manifest_path")
            if raw:
                p = resolve_path(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append("step135_pointer_manifest_missing")
            else:
                warnings.append("step135_pointer_has_no_manifest_path")
        except Exception as exc:
            warnings.append(f"step135_pointer_unreadable: {exc}")
    else:
        warnings.append("step135_pointer_missing")

    candidates = sorted(
        [p for p in project_root.glob(STEP135_FALLBACK_GLOB) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        warnings.append("using_step135_fallback_manifest_search")
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


def check_step135_manifest(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "violations": [],
        "warnings": [],
        "summary": {},
    }

    if path is None or not path.exists():
        result["violations"].append("step135_manifest_missing")
        return result

    try:
        payload = load_json_any(path)
    except Exception as exc:
        result["violations"].append(f"step135_manifest_unreadable: {exc}")
        return result

    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")
    result["readiness_lock_status"] = payload.get("readiness_lock_status")
    result["actual_execution_allowed"] = payload.get("actual_execution_allowed")
    result["actual_results"] = payload.get("actual_results")
    result["release_requirements"] = payload.get("release_requirements", [])
    result["summary"] = {
        "audit_status": payload.get("audit_status"),
        "readiness_lock_status": payload.get("readiness_lock_status"),
        "actual_execution_allowed": payload.get("actual_execution_allowed"),
        "actual_results": payload.get("actual_results"),
    }

    if payload.get("step") != 135:
        result["violations"].append(f"step135_manifest_step_mismatch: {payload.get('step')}")

    if payload.get("audit_status") != "PASS":
        result["violations"].append(f"step135_audit_not_pass: {payload.get('audit_status')}")

    if payload.get("readiness_lock_status") != "LOCKED_READY_FOR_EXPLICIT_RELEASE":
        result["violations"].append(
            f"step135_lock_status_not_ready_for_release: {payload.get('readiness_lock_status')}"
        )

    result["violations"].extend(recursive_forbidden_true(payload))

    release_requirements = payload.get("release_requirements", [])
    if not isinstance(release_requirements, list) or not release_requirements:
        result["violations"].append("step135_release_requirements_missing")

    return result


def build_operator_checklist() -> List[Dict[str, Any]]:
    return [
        {
            "item_id": "OP-001",
            "name": "Confirm Step 135 readiness lock is PASS",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
        {
            "item_id": "OP-002",
            "name": "Confirm actual reward ablation result path is empty or archived",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
        {
            "item_id": "OP-003",
            "name": "Confirm run matrix is A x R0-R5 x seeds 1-3 only",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
        {
            "item_id": "OP-004",
            "name": "Confirm actual runner code has been reviewed before enabling actual mode",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
        {
            "item_id": "OP-005",
            "name": "Confirm no winner or trainable reward promotion will be made in the release step",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
        {
            "item_id": "OP-006",
            "name": "Confirm post-execution result ingestion will run before selection or promotion",
            "required": True,
            "operator_must_confirm": True,
            "confirmed": False,
        },
    ]


def build_release_request_markdown(payload: Dict[str, Any]) -> str:
    matrix = payload["requested_matrix"]
    lines = [
        "# Step 136 reward ablation actual execution release request package",
        "",
        "This document is a release-request package, not an execution approval.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- release_request_status: `{payload['release_request_status']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- train_with_this_reward_allowed: `{payload['train_with_this_reward_allowed']}`",
        "",
        "## Requested matrix",
        "",
        f"- conditions: `{matrix['conditions']}`",
        f"- reward_ids: `{matrix['reward_ids']}`",
        f"- seeds: `{matrix['seeds']}`",
        f"- planned_run_count: `{matrix['planned_run_count']}`",
        "",
        "## Operator checklist",
        "",
    ]

    for item in payload["operator_checklist"]:
        lines.append(
            f"- [ ] {item['item_id']} - {item['name']} "
            f"(required={item['required']}, confirmed={item['confirmed']})"
        )

    lines.extend([
        "",
        "## Guard statement",
        "",
        "Step 136 does not release actual execution. It only creates the package required for a future explicit release step.",
        "Actual execution remains disabled until a separate release manifest is created and validated.",
        "",
    ])

    return "\n".join(lines)


def run_release_request(project_root: Path, output_root: Path, release_reason: str) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step135_manifest, find_warnings = find_step135_manifest(project_root)
    step135_check = check_step135_manifest(step135_manifest)

    blocking_reasons: List[str] = []
    warnings: List[str] = []
    warnings.extend(find_warnings)
    warnings.extend(step135_check.get("warnings", []))
    blocking_reasons.extend(step135_check.get("violations", []))

    if not release_reason.strip():
        blocking_reasons.append("release_reason_required")

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"

    release_request_status = (
        "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL"
        if audit_status == "PASS"
        else "REQUEST_PACKAGE_BLOCKED"
    )

    manifest_path = output_root / "reward_ablation_actual_execution_release_request_step136_manifest.json"
    md_path = output_root / "reward_ablation_actual_execution_release_request_step136.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "release_request_status": release_request_status,
        "release_reason": release_reason,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step135_manifest": str(step135_manifest) if step135_manifest else "",
        "step135_check": step135_check,
        "requested_matrix": EXPECTED_MATRIX,
        "operator_checklist": build_operator_checklist(),
        "release_manifest_required_next": True,
        "next_step_recommendation": "Step 137 explicit release manifest validator or Step 137 actual runner implementation review",
        "actual_execution_allowed": False,
        "actual_execution_released": False,
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
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
        "scope_note": (
            "Step 136 creates an explicit release-request package. It does not unlock actual execution, "
            "does not run reward ablation, does not write result rows, does not select a winner, and does not promote a reward."
        ),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_release_request_markdown(payload))

    latest_path = project_root / "05_training" / "rewards" / "reward_ablation_actual_execution_release_request_step136.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
        "audit_status": audit_status,
        "release_request_status": release_request_status,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    source_md = project_root / "05_training" / "rewards" / "reward_ablation_actual_execution_release_request_step136.md"
    dump_text(source_md, build_release_request_markdown(payload))

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_execution_release_request_step136")
    parser.add_argument(
        "--release-reason",
        default="Prepare explicit release request package after Step 135 readiness lock.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_release_request(project_root, output_root, args.release_reason)

    print("[OK] Step 136 reward ablation actual execution release request package completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] release_request_status    : {payload['release_request_status']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest                  : {payload['manifest_path']}")
    print(f"[OK] markdown                  : {payload['markdown_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 136 release request package failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
