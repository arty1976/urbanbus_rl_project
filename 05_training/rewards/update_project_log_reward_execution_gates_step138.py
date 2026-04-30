from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 138
ARTIFACT_VERSION = "reward_execution_gate_project_log_update_step138_v1"

STEP_POINTERS = [
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
    {
        "name": "step135_readiness_lock",
        "step": 135,
        "pointer": "05_training/rewards/reward_ablation_actual_execution_readiness_lock_step135.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step135*manifest*.json",
    },
    {
        "name": "step136_release_request_package",
        "step": 136,
        "pointer": "05_training/rewards/reward_ablation_actual_execution_release_request_step136.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step136*manifest*.json",
    },
    {
        "name": "step137_runner_implementation_review",
        "step": 137,
        "pointer": "05_training/rewards/actual_reward_ablation_runner_implementation_review_step137.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step137*manifest*.json",
    },
]

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

SECTION_START = "<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_START -->"
SECTION_END = "<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = project_root / p
    return p


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def find_manifest(project_root: Path, pointer_rel: str, fallback_glob: str) -> Tuple[Optional[Path], List[str]]:
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

    # Step 138 patch:
    # Step 137 static_review.required_source_tokens is a token-presence map.
    # A value True there means "the source code contains this safety token",
    # not "the experiment flag is true". Therefore it must not be treated as
    # a forbidden execution/result/claim flag.
    ignored_token_presence_maps = {
        "required_source_tokens",
    }

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)

            if key in ignored_token_presence_maps:
                continue

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


def check_step_manifest(project_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    manifest, warnings = find_manifest(project_root, spec["pointer"], spec["fallback_glob"])

    result: Dict[str, Any] = {
        "name": spec["name"],
        "required_step": spec["step"],
        "pointer": spec["pointer"],
        "manifest_path": str(manifest) if manifest else "",
        "exists": bool(manifest and manifest.exists()),
        "warnings": warnings,
        "violations": [],
        "summary": {},
    }

    if manifest is None or not manifest.exists():
        result["violations"].append(f"manifest_missing: step{spec['step']}")
        return result

    try:
        payload = load_json_any(manifest)
    except Exception as exc:
        result["violations"].append(f"manifest_unreadable: step{spec['step']}: {exc}")
        return result

    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")
    result["review_status"] = payload.get("review_status")
    result["release_request_status"] = payload.get("release_request_status")
    result["readiness_lock_status"] = payload.get("readiness_lock_status")
    result["summary"] = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}

    if payload.get("step") != spec["step"]:
        result["violations"].append(
            f"step_mismatch: expected={spec['step']} got={payload.get('step')}"
        )

    if payload.get("audit_status") != "PASS":
        result["violations"].append(f"audit_not_pass: step{spec['step']}: {payload.get('audit_status')}")

    result["violations"].extend(recursive_forbidden_true(payload))

    return result


def build_project_log_section(created_at_utc: str, checks: List[Dict[str, Any]]) -> str:
    lines = [
        SECTION_START,
        "",
        "## Step 138 - Reward execution gate project log update",
        "",
        f"Created at UTC: `{created_at_utc}`",
        "",
        "### Scope",
        "",
        "Step 138 records the reward ablation execution gate status after Step 131 through Step 137.",
        "This is a project-log update only. It does not run actual reward ablation.",
        "",
        "### Completed gate chain",
        "",
        "- Step 131: actual reward ablation execution environment preflight",
        "- Step 132: reward ablation command dry-run executor",
        "- Step 133: actual reward ablation runner guard",
        "- Step 134: command-to-guarded-runner integration",
        "- Step 135: actual execution readiness lock",
        "- Step 136: actual execution release request package",
        "- Step 137: actual runner implementation review gate",
        "",
        "### Manifest status summary",
        "",
        "| Step | Name | Audit status | Extra status | Manifest |",
        "|---:|---|---|---|---|",
    ]

    for check in checks:
        step = check.get("required_step")
        name = check.get("name")
        audit = check.get("audit_status", "")
        extra = (
            check.get("readiness_lock_status")
            or check.get("release_request_status")
            or check.get("review_status")
            or ""
        )
        manifest = check.get("manifest_path", "")
        lines.append(f"| {step} | {name} | {audit} | {extra} | `{manifest}` |")

    lines.extend([
        "",
        "### Locked guard state",
        "",
        "- actual_execution_allowed = false",
        "- actual_execution_released = false",
        "- actual_executed = false",
        "- actual_results = false",
        "- reward_result_written = false",
        "- winner_selected = false",
        "- trainable_reward_promoted = false",
        "- train_with_this_reward_allowed = false",
        "- actual_training_allowed = false",
        "- final_reward_design_claim_allowed = false",
        "- best_reward_claim_allowed = false",
        "- paper_level_claim_allowed = false",
        "- causal_performance_claim_allowed = false",
        "",
        "### Interpretation",
        "",
        "The reward ablation execution path is prepared through dry-run commands, guarded runner checks, readiness lock, release request packaging, and runner implementation review.",
        "However, actual execution remains locked. No reward ablation result exists yet, no winner has been selected, no reward has been promoted for MAPPO training, and no paper-level or causal-performance claim is allowed.",
        "",
        "### Recommended next step",
        "",
        "Step 139 should either update the project log commit/push status or define an explicit release-manifest design only if the operator is ready to unlock actual execution in a separate controlled step.",
        "",
        SECTION_END,
        "",
    ])

    return "\n".join(lines)


def upsert_section(existing: str, section: str) -> str:
    if SECTION_START in existing and SECTION_END in existing:
        start = existing.index(SECTION_START)
        end = existing.index(SECTION_END) + len(SECTION_END)
        return existing[:start].rstrip() + "\n\n" + section.strip() + "\n\n" + existing[end:].lstrip()

    if not existing.endswith("\n"):
        existing += "\n"

    return existing.rstrip() + "\n\n" + section.strip() + "\n"


def run_update(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    checks = [check_step_manifest(project_root, spec) for spec in STEP_POINTERS]

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for check in checks:
        blocking_reasons.extend(check.get("violations", []))
        warnings.extend(check.get("warnings", []))

    project_log = project_root / "project_log.md"
    if not project_log.exists():
        blocking_reasons.append("project_log_missing")

    created_at = utc_now()
    section = build_project_log_section(created_at, checks)

    if project_log.exists() and not blocking_reasons:
        existing = read_text_any(project_log)
        updated = upsert_section(existing, section)
        dump_text(project_log, updated)

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": created_at,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "project_log": str(project_log),
        "project_log_updated": bool(audit_status == "PASS"),
        "step_checks": checks,
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
        "scope_note": (
            "Step 138 updates project_log.md with the reward execution gate status. "
            "It does not run actual ablation and does not unlock execution."
        ),
    }

    manifest_path = output_root / "reward_execution_gate_project_log_update_step138_manifest.json"
    dump_json(manifest_path, payload)

    status_path = project_root / "05_training" / "rewards" / "reward_execution_gate_project_log_update_step138.json"
    dump_json(status_path, {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": created_at,
        "audit_status": audit_status,
        "project_log_updated": bool(audit_status == "PASS"),
        "manifest_path": str(manifest_path),
        "actual_execution_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    })

    latest_path = project_root / "05_training" / "rewards" / "reward_execution_gate_project_log_update_step138.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "audit_status": audit_status,
        "project_log_updated": bool(audit_status == "PASS"),
        "actual_execution_allowed": False,
        "actual_results": False,
        "created_at_utc": created_at,
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_execution_gate_project_log_update_step138")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_update(project_root, output_root)

    print("[OK] Step 138 reward execution gate project log update completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] project_log_updated       : {payload['project_log_updated']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 138 project log update failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
