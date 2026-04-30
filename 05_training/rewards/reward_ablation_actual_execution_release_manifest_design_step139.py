from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 139
ARTIFACT_VERSION = "reward_ablation_actual_execution_release_manifest_design_step139_v1"

UPSTREAM_SPECS = [
    {
        "name": "step135_readiness_lock",
        "step": 135,
        "pointer": "05_training/rewards/reward_ablation_actual_execution_readiness_lock_step135.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step135*manifest*.json",
        "expected_status_key": "readiness_lock_status",
        "expected_status_value": "LOCKED_READY_FOR_EXPLICIT_RELEASE",
    },
    {
        "name": "step136_release_request_package",
        "step": 136,
        "pointer": "05_training/rewards/reward_ablation_actual_execution_release_request_step136.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step136*manifest*.json",
        "expected_status_key": "release_request_status",
        "expected_status_value": "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL",
    },
    {
        "name": "step137_runner_review",
        "step": 137,
        "pointer": "05_training/rewards/actual_reward_ablation_runner_implementation_review_step137.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step137*manifest*.json",
        "expected_status_key": "review_status",
        "expected_status_value": "RUNNER_GUARD_REVIEW_PASS_ACTUAL_STILL_LOCKED",
    },
    {
        "name": "step138_project_log_update",
        "step": 138,
        "pointer": "05_training/rewards/reward_execution_gate_project_log_update_step138.latest.json",
        "fallback_glob": "artifacts/rewards/**/*step138*manifest*.json",
        "expected_status_key": "project_log_updated",
        "expected_status_value": True,
    },
]

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


def resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = project_root / p
    return p


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
    ignored_token_presence_maps = {"required_source_tokens"}

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


def check_upstream_manifest(project_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path, warnings = find_manifest(project_root, spec["pointer"], spec["fallback_glob"])
    result: Dict[str, Any] = {
        "name": spec["name"],
        "required_step": spec["step"],
        "pointer": spec["pointer"],
        "manifest_path": str(manifest_path) if manifest_path else "",
        "exists": bool(manifest_path and manifest_path.exists()),
        "warnings": warnings,
        "violations": [],
    }

    if manifest_path is None or not manifest_path.exists():
        result["violations"].append(f"manifest_missing: {spec['name']}")
        return result

    try:
        payload = load_json_any(manifest_path)
    except Exception as exc:
        result["violations"].append(f"manifest_unreadable: {spec['name']}: {exc}")
        return result

    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")
    result[spec["expected_status_key"]] = payload.get(spec["expected_status_key"])

    if payload.get("step") != spec["step"]:
        result["violations"].append(
            f"step_mismatch: {spec['name']}: expected={spec['step']} got={payload.get('step')}"
        )
    if payload.get("audit_status") != "PASS":
        result["violations"].append(f"audit_not_pass: {spec['name']}: {payload.get('audit_status')}")

    actual_value = payload.get(spec["expected_status_key"])
    if actual_value != spec["expected_status_value"]:
        result["violations"].append(
            f"expected_status_mismatch: {spec['name']}: "
            f"{spec['expected_status_key']} expected={spec['expected_status_value']} got={actual_value}"
        )

    result["violations"].extend(recursive_forbidden_true(payload))
    return result


def build_release_manifest_template(upstream_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "release_manifest_version": "actual_reward_ablation_release_manifest_v1",
        "created_from_step": 139,
        "purpose": "Template only. This file is not an execution release.",
        "operator_fields": {
            "requested_by": "",
            "reviewed_by": "",
            "approved_by": "",
            "approval_timestamp_utc": "",
            "operator_approval_phrase": "",
        },
        "required_operator_approval_phrase": "I APPROVE ACTUAL REWARD ABLATION EXECUTION FOR A_R0_TO_R5_SEEDS_1_TO_3",
        "requested_matrix": EXPECTED_MATRIX,
        "upstream_manifest_paths": {check["name"]: check.get("manifest_path", "") for check in upstream_checks},
        "approval_gates": {
            "step135_lock_passed": False,
            "step136_release_request_reviewed": False,
            "step137_runner_review_passed": False,
            "step138_project_log_updated": False,
            "operator_explicit_approval": False,
            "result_output_path_confirmed_empty_or_archived": False,
            "no_winner_selection_in_release_step": False,
            "post_execution_ingestion_step_required": True,
        },
        "release_decision": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "release_ready": False,
            "release_blocking_reason": "template_created_by_step139_not_operator_approved",
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 139 reward ablation actual execution release manifest design",
        "",
        "This step designs the release manifest schema. It does not release actual execution.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- design_status: `{payload['design_status']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        "",
        "## Designed matrix",
        "",
        "- condition: `A`",
        "- reward candidates: `R0` through `R5`",
        "- seeds: `1, 2, 3`",
        "- planned run count: `18`",
        "",
        "## Next step",
        "",
        "Step 140 may create a concrete release manifest only if the operator explicitly approves the actual execution.",
        "Winner selection and reward promotion must remain separate post-result-ingestion steps.",
        "",
    ])


def run_design(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    upstream_checks = [check_upstream_manifest(project_root, spec) for spec in UPSTREAM_SPECS]
    blocking_reasons: List[str] = []
    warnings: List[str] = []
    for check in upstream_checks:
        blocking_reasons.extend(check.get("violations", []))
        warnings.extend(check.get("warnings", []))

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    design_status = (
        "RELEASE_MANIFEST_SCHEMA_DESIGNED_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "RELEASE_MANIFEST_SCHEMA_DESIGN_BLOCKED"
    )

    release_template = build_release_manifest_template(upstream_checks)
    manifest_path = output_root / "reward_ablation_actual_execution_release_manifest_design_step139_manifest.json"
    template_path = output_root / "actual_reward_ablation_release_manifest_template_step139.json"
    md_path = output_root / "reward_ablation_actual_execution_release_manifest_design_step139.md"

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
        "design_status": design_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "upstream_checks": upstream_checks,
        "release_manifest_template_path": str(template_path),
        "requested_matrix": EXPECTED_MATRIX,
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
        "next_step_recommendation": "Step 140 may instantiate a concrete release manifest after explicit operator approval.",
        "scope_note": "Step 139 only designs the release manifest schema and template. It does not unlock actual execution.",
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_json(template_path, release_template)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "reward_ablation_actual_execution_release_manifest_design_step139.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "reward_ablation_actual_execution_release_manifest_design_step139.latest.json", {
        "manifest_path": str(manifest_path),
        "template_path": str(template_path),
        "audit_status": audit_status,
        "design_status": design_status,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_execution_release_manifest_design_step139")
    args = parser.parse_args()

    payload = run_design(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 139 reward ablation actual execution release manifest design completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] design_status             : {payload['design_status']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] template                  : {payload['release_manifest_template_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 139 release manifest design failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
