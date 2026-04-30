from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 137
ARTIFACT_VERSION = "actual_reward_ablation_runner_implementation_review_step137_v1"

RUNNER_REL = "05_training/rewards/run_actual_reward_ablation_candidate.py"
STEP136_POINTER = "05_training/rewards/reward_ablation_actual_execution_release_request_step136.latest.json"
STEP136_FALLBACK_GLOB = "artifacts/rewards/**/*step136*manifest*.json"

REQUIRED_SOURCE_TOKENS = [
    "actual_execution_blocked_by_step133_guard",
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

REQUIRED_BLOCKING_TOKENS_ANY = [
    "mode == \"actual\"",
    "mode == 'actual'",
    "execute_actual",
]

FORBIDDEN_TRUE_SNIPPETS = [
    '"actual_executed": True',
    '"actual_results": True',
    '"reward_result_written": True',
    '"winner_selected": True',
    '"trainable_reward_promoted": True',
    '"train_with_this_reward_allowed": True',
    '"actual_training_allowed": True',
    '"paper_level_claim_allowed": True',
    '"causal_performance_claim_allowed": True',
    "'actual_executed': True",
    "'actual_results': True",
    "'reward_result_written': True",
    "'winner_selected': True",
    "'trainable_reward_promoted': True",
    "'train_with_this_reward_allowed': True",
    "'actual_training_allowed': True",
    "'paper_level_claim_allowed': True",
    "'causal_performance_claim_allowed': True",
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
    raise RuntimeError(f"failed_to_read_text: {path}")


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


def find_step136_manifest(project_root: Path) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP136_POINTER

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            raw = payload.get("manifest_path") or payload.get("path") or payload.get("latest_manifest_path")
            if raw:
                p = resolve_path(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append("step136_pointer_manifest_missing")
            else:
                warnings.append("step136_pointer_has_no_manifest_path")
        except Exception as exc:
            warnings.append(f"step136_pointer_unreadable: {exc}")
    else:
        warnings.append("step136_pointer_missing")

    candidates = sorted(
        [p for p in project_root.glob(STEP136_FALLBACK_GLOB) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        warnings.append("using_step136_fallback_manifest_search")
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


def check_step136_manifest(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "violations": [],
        "warnings": [],
    }

    if path is None or not path.exists():
        result["violations"].append("step136_manifest_missing")
        return result

    try:
        payload = load_json_any(path)
    except Exception as exc:
        result["violations"].append(f"step136_manifest_unreadable: {exc}")
        return result

    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")
    result["release_request_status"] = payload.get("release_request_status")
    result["actual_execution_allowed"] = payload.get("actual_execution_allowed")
    result["actual_execution_released"] = payload.get("actual_execution_released")

    if payload.get("step") != 136:
        result["violations"].append(f"step136_manifest_step_mismatch: {payload.get('step')}")

    if payload.get("audit_status") != "PASS":
        result["violations"].append(f"step136_audit_not_pass: {payload.get('audit_status')}")

    if payload.get("release_request_status") != "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL":
        result["violations"].append(
            f"step136_release_request_status_invalid: {payload.get('release_request_status')}"
        )

    result["violations"].extend(recursive_forbidden_true(payload))

    return result


def static_review_runner_source(project_root: Path) -> Dict[str, Any]:
    runner_path = project_root / RUNNER_REL
    result: Dict[str, Any] = {
        "runner_path": str(runner_path),
        "exists": runner_path.exists(),
        "violations": [],
        "warnings": [],
        "required_source_tokens": {},
        "forbidden_true_snippets_found": [],
    }

    if not runner_path.exists():
        result["violations"].append("runner_file_missing")
        return result

    try:
        text = read_text_any(runner_path)
    except Exception as exc:
        result["violations"].append(f"runner_file_unreadable: {exc}")
        return result

    for token in REQUIRED_SOURCE_TOKENS:
        present = token in text
        result["required_source_tokens"][token] = present
        if not present:
            result["violations"].append(f"required_source_token_missing: {token}")

    if not any(token in text for token in REQUIRED_BLOCKING_TOKENS_ANY):
        result["violations"].append("actual_mode_blocking_logic_token_missing")

    for snippet in FORBIDDEN_TRUE_SNIPPETS:
        if snippet in text:
            result["forbidden_true_snippets_found"].append(snippet)
            result["violations"].append(f"forbidden_true_snippet_found: {snippet}")

    return result


def load_manifest_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return load_json_any(path)


def check_no_true_flags(payload: Dict[str, Any], prefix: str) -> List[str]:
    out: List[str] = []
    for key in [
        "actual_executed",
        "actual_results",
        "reward_result_written",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "actual_training_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        if bool(payload.get(key, False)):
            out.append(f"{prefix}_forbidden_true: {key}")
    return out


def dynamic_review_runner(project_root: Path, output_root: Path) -> Dict[str, Any]:
    runner_path = project_root / RUNNER_REL
    result: Dict[str, Any] = {
        "runner_path": str(runner_path),
        "dry_run": {},
        "actual_attempt": {},
        "violations": [],
        "warnings": [],
    }

    if not runner_path.exists():
        result["violations"].append("runner_file_missing_for_dynamic_review")
        return result

    dry_root = output_root / "dynamic_dry_run_R0_seed001"
    actual_root = output_root / "dynamic_actual_attempt_R0_seed001"

    dry_cmd = [
        sys.executable,
        str(runner_path),
        "--project-root",
        str(project_root),
        "--reward-id",
        "R0",
        "--seed",
        "1",
        "--condition",
        "A",
        "--mode",
        "dry-run",
        "--output-root",
        safe_rel(dry_root, project_root),
        "--candidate-label",
        "step137_dynamic_dry",
    ]

    dry_proc = subprocess.run(
        dry_cmd,
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dry_manifest_path = dry_root / "actual_reward_ablation_runner_guard_step133_manifest.json"

    result["dry_run"] = {
        "cmd": dry_cmd,
        "returncode": int(dry_proc.returncode),
        "stdout_tail": dry_proc.stdout[-2000:],
        "stderr_tail": dry_proc.stderr[-2000:],
        "manifest_path": str(dry_manifest_path),
        "manifest_exists": dry_manifest_path.exists(),
    }

    if dry_proc.returncode != 0:
        result["violations"].append("dry_run_returned_nonzero")

    dry_manifest = load_manifest_if_exists(dry_manifest_path)
    if dry_manifest is None:
        result["violations"].append("dry_run_manifest_missing")
    else:
        result["dry_run"]["audit_status"] = dry_manifest.get("audit_status")
        if dry_manifest.get("audit_status") != "PASS":
            result["violations"].append(f"dry_run_manifest_not_pass: {dry_manifest.get('audit_status')}")
        result["violations"].extend(check_no_true_flags(dry_manifest, "dry_run_manifest"))

    actual_cmd = [
        sys.executable,
        str(runner_path),
        "--project-root",
        str(project_root),
        "--reward-id",
        "R0",
        "--seed",
        "1",
        "--condition",
        "A",
        "--mode",
        "actual",
        "--output-root",
        safe_rel(actual_root, project_root),
        "--candidate-label",
        "step137_dynamic_actual_block",
    ]

    actual_proc = subprocess.run(
        actual_cmd,
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual_manifest_path = actual_root / "actual_reward_ablation_runner_guard_step133_manifest.json"

    result["actual_attempt"] = {
        "cmd": actual_cmd,
        "returncode": int(actual_proc.returncode),
        "stdout_tail": actual_proc.stdout[-2000:],
        "stderr_tail": actual_proc.stderr[-2000:],
        "manifest_path": str(actual_manifest_path),
        "manifest_exists": actual_manifest_path.exists(),
    }

    if actual_proc.returncode == 0:
        result["violations"].append("actual_attempt_unexpectedly_returned_zero")

    actual_manifest = load_manifest_if_exists(actual_manifest_path)
    if actual_manifest is None:
        result["violations"].append("actual_attempt_manifest_missing")
    else:
        result["actual_attempt"]["audit_status"] = actual_manifest.get("audit_status")
        result["actual_attempt"]["blocking_reasons"] = actual_manifest.get("blocking_reasons", [])
        if actual_manifest.get("audit_status") != "BLOCKED":
            result["violations"].append(
                f"actual_attempt_manifest_not_blocked: {actual_manifest.get('audit_status')}"
            )
        if "actual_execution_blocked_by_step133_guard" not in actual_manifest.get("blocking_reasons", []):
            result["violations"].append("actual_attempt_missing_expected_blocking_reason")
        result["violations"].extend(check_no_true_flags(actual_manifest, "actual_attempt_manifest"))

    return result


def build_review_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Step 137 actual reward ablation runner implementation review gate",
        "",
        "This review gate checks the current guarded runner implementation before any future execution-release manifest.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- review_status: `{payload['review_status']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_executed: `{payload['actual_executed']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        "",
        "## Review summary",
        "",
        f"- static violations: `{len(payload['static_review'].get('violations', []))}`",
        f"- dynamic violations: `{len(payload['dynamic_review'].get('violations', []))}`",
        "",
        "## Guard statement",
        "",
        "The runner must continue to block actual mode until a separate reviewed release mechanism exists.",
        "Step 137 does not unlock execution.",
        "",
    ]
    return "\n".join(lines)


def run_review(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step136_manifest, find_warnings = find_step136_manifest(project_root)
    step136_check = check_step136_manifest(step136_manifest)
    static_review = static_review_runner_source(project_root)
    dynamic_review = dynamic_review_runner(project_root, output_root / "dynamic_review")

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    warnings.extend(find_warnings)
    warnings.extend(step136_check.get("warnings", []))
    blocking_reasons.extend(step136_check.get("violations", []))
    blocking_reasons.extend(static_review.get("violations", []))
    blocking_reasons.extend(dynamic_review.get("violations", []))

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    review_status = (
        "RUNNER_GUARD_REVIEW_PASS_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "RUNNER_GUARD_REVIEW_BLOCKED"
    )

    manifest_path = output_root / "actual_reward_ablation_runner_implementation_review_step137_manifest.json"
    md_path = output_root / "actual_reward_ablation_runner_implementation_review_step137.md"

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
        "review_status": review_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step136_manifest": str(step136_manifest) if step136_manifest else "",
        "step136_check": step136_check,
        "static_review": static_review,
        "dynamic_review": dynamic_review,
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
        "next_step_recommendation": "Step 138 project log update for reward ablation execution gates, or Step 138 explicit release manifest design if the operator is ready.",
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
        "scope_note": (
            "Step 137 reviews the guarded runner implementation. It intentionally confirms that actual execution remains blocked."
        ),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_review_markdown(payload))

    latest_path = project_root / "05_training" / "rewards" / "actual_reward_ablation_runner_implementation_review_step137.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
        "audit_status": audit_status,
        "review_status": review_status,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    source_md = project_root / "05_training" / "rewards" / "actual_reward_ablation_runner_implementation_review_step137.md"
    dump_text(source_md, build_review_markdown(payload))

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/actual_reward_ablation_runner_implementation_review_step137")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_review(project_root, output_root)

    print("[OK] Step 137 actual reward ablation runner implementation review completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] review_status             : {payload['review_status']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest                  : {payload['manifest_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 137 runner implementation review failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
