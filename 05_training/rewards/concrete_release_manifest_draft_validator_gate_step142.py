from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 142
ARTIFACT_VERSION = "concrete_release_manifest_draft_validator_gate_step142_v1"

STEP140_POINTER = "05_training/rewards/concrete_reward_ablation_release_manifest_draft_step140.latest.json"
STEP141_POINTER = "05_training/rewards/baseline_reference_manifest_step141.latest.json"

STEP140_FALLBACK_GLOB = "artifacts/rewards/concrete_reward_ablation_release_manifest_draft_step140/concrete_reward_ablation_release_manifest_draft_step140_manifest.json"
STEP141_FALLBACK_GLOB = "artifacts/rewards/baseline_reference_manifest_step141/baseline_reference_manifest_step141.json"

EXPECTED_A_RELEASE_MATRIX = {
    "conditions": ["A"],
    "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
    "seeds": [1, 2, 3],
    "planned_run_count": 18,
}

EXPECTED_BASELINES = {"B0R", "B1", "B2"}

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


def find_manifest(
    project_root: Path,
    pointer_rel: str,
    fallback_rel: str,
    pointer_keys: List[str],
) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / pointer_rel

    if pointer.exists():
        try:
            pointer_payload = load_json_any(pointer)
            for key in pointer_keys:
                raw = pointer_payload.get(key)
                if raw:
                    p = resolve_path(project_root, str(raw))
                    if p.exists():
                        return p, warnings
                    warnings.append(f"pointer_{key}_missing: {p}")
            warnings.append(f"pointer_has_no_known_manifest_key: {pointer_rel}")
        except Exception as exc:
            warnings.append(f"pointer_unreadable: {pointer_rel}: {exc}")
    else:
        warnings.append(f"pointer_missing: {pointer_rel}")

    fallback = project_root / fallback_rel
    if fallback.exists():
        warnings.append(f"using_fallback_manifest: {fallback_rel}")
        return fallback, warnings

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


def validate_step140(project_root: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    violations: List[str] = []

    manifest_path, find_warnings = find_manifest(
        project_root,
        STEP140_POINTER,
        STEP140_FALLBACK_GLOB,
        ["manifest_path"],
    )
    warnings.extend(find_warnings)

    result: Dict[str, Any] = {
        "component": "step140_concrete_release_manifest_draft",
        "manifest_path": str(manifest_path) if manifest_path else "",
        "draft_path": "",
        "exists": bool(manifest_path and manifest_path.exists()),
        "warnings": warnings,
        "violations": violations,
    }

    if manifest_path is None or not manifest_path.exists():
        violations.append("step140_manifest_missing")
        return result

    try:
        manifest = load_json_any(manifest_path)
    except Exception as exc:
        violations.append(f"step140_manifest_unreadable: {exc}")
        return result

    result["manifest_summary"] = {
        "step": manifest.get("step"),
        "artifact_version": manifest.get("artifact_version"),
        "audit_status": manifest.get("audit_status"),
        "draft_status": manifest.get("draft_status"),
        "release_validation_status": manifest.get("release_validation_status"),
        "actual_execution_allowed": bool(manifest.get("actual_execution_allowed", False)),
        "actual_execution_released": bool(manifest.get("actual_execution_released", False)),
        "actual_results": bool(manifest.get("actual_results", False)),
        "winner_selected": bool(manifest.get("winner_selected", False)),
    }

    if manifest.get("step") != 140:
        violations.append(f"step140_step_mismatch: {manifest.get('step')}")
    if manifest.get("audit_status") != "PASS":
        violations.append(f"step140_audit_not_pass: {manifest.get('audit_status')}")
    if manifest.get("draft_status") != "CONCRETE_RELEASE_MANIFEST_DRAFT_CREATED_ACTUAL_STILL_LOCKED":
        violations.append(f"step140_draft_status_invalid: {manifest.get('draft_status')}")
    if manifest.get("release_validation_status") not in {
        "BLOCKED_PENDING_OPERATOR_APPROVAL",
        "DRAFT_READY_FOR_SEPARATE_MANUAL_RELEASE_REVIEW",
    }:
        violations.append(f"step140_release_validation_status_invalid: {manifest.get('release_validation_status')}")

    violations.extend([f"manifest.{v}" for v in recursive_forbidden_true(manifest)])

    raw_draft = manifest.get("concrete_draft_path")
    if not raw_draft:
        violations.append("step140_concrete_draft_path_missing")
        return result

    draft_path = resolve_path(project_root, str(raw_draft))
    result["draft_path"] = str(draft_path)

    if not draft_path.exists():
        violations.append(f"step140_concrete_draft_missing: {draft_path}")
        return result

    try:
        draft = load_json_any(draft_path)
    except Exception as exc:
        violations.append(f"step140_concrete_draft_unreadable: {exc}")
        return result

    result["draft_summary"] = {
        "release_manifest_version": draft.get("release_manifest_version"),
        "created_from_step": draft.get("created_from_step"),
        "release_validation_status": draft.get("release_decision", {}).get("release_validation_status"),
        "operator_fields_complete": bool(draft.get("operator_fields_complete", False)),
        "operator_approval_phrase_matches": bool(draft.get("operator_approval_phrase_matches", False)),
        "requested_matrix": draft.get("requested_matrix", {}),
    }

    if draft.get("release_manifest_version") != "actual_reward_ablation_release_manifest_v1":
        violations.append("draft_release_manifest_version_mismatch")
    if draft.get("created_from_step") != 140:
        violations.append(f"draft_created_from_step_mismatch: {draft.get('created_from_step')}")
    if draft.get("requested_matrix") != EXPECTED_A_RELEASE_MATRIX:
        violations.append(f"draft_requested_matrix_mismatch: {draft.get('requested_matrix')}")

    decision = draft.get("release_decision", {})
    if bool(decision.get("actual_execution_allowed", False)):
        violations.append("draft_actual_execution_allowed_true")
    if bool(decision.get("actual_execution_released", False)):
        violations.append("draft_actual_execution_released_true")
    if bool(decision.get("release_ready", False)):
        violations.append("draft_release_ready_true")

    if decision.get("release_validation_status") not in {
        "BLOCKED_PENDING_OPERATOR_APPROVAL",
        "DRAFT_READY_FOR_SEPARATE_MANUAL_RELEASE_REVIEW",
    }:
        violations.append(f"draft_release_validation_status_invalid: {decision.get('release_validation_status')}")

    non_claim_guards = draft.get("non_claim_guards", {})
    if not isinstance(non_claim_guards, dict) or not non_claim_guards:
        violations.append("draft_non_claim_guards_missing")
    else:
        for key, value in non_claim_guards.items():
            if bool(value):
                violations.append(f"draft_non_claim_guard_true: {key}")

    violations.extend([f"draft.{v}" for v in recursive_forbidden_true(draft)])
    return result


def validate_step141(project_root: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    violations: List[str] = []

    manifest_path, find_warnings = find_manifest(
        project_root,
        STEP141_POINTER,
        STEP141_FALLBACK_GLOB,
        ["manifest_path"],
    )
    warnings.extend(find_warnings)

    result: Dict[str, Any] = {
        "component": "step141_b_group_baseline_reference",
        "manifest_path": str(manifest_path) if manifest_path else "",
        "exists": bool(manifest_path and manifest_path.exists()),
        "warnings": warnings,
        "violations": violations,
    }

    if manifest_path is None or not manifest_path.exists():
        violations.append("step141_manifest_missing")
        return result

    try:
        manifest = load_json_any(manifest_path)
    except Exception as exc:
        violations.append(f"step141_manifest_unreadable: {exc}")
        return result

    refs = manifest.get("baseline_references", [])
    baseline_ids = {str(r.get("baseline_id")) for r in refs}
    ready_refs = [
        r for r in refs
        if r.get("reference_status") == "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    ]

    result["manifest_summary"] = {
        "step": manifest.get("step"),
        "artifact_version": manifest.get("artifact_version"),
        "audit_status": manifest.get("audit_status"),
        "baseline_reference_status": manifest.get("baseline_reference_status"),
        "ready_baseline_count": manifest.get("ready_baseline_count"),
        "required_baseline_count": manifest.get("required_baseline_count"),
        "baseline_ids": sorted(baseline_ids),
        "causal_comparison_allowed": bool(manifest.get("causal_comparison_allowed", False)),
        "paper_level_claim_allowed": bool(manifest.get("paper_level_claim_allowed", False)),
    }

    if manifest.get("step") != 141:
        violations.append(f"step141_step_mismatch: {manifest.get('step')}")
    if manifest.get("audit_status") != "PASS":
        violations.append(f"step141_audit_not_pass: {manifest.get('audit_status')}")
    if manifest.get("baseline_reference_status") != "B_GROUP_BASELINE_REFERENCE_READY":
        violations.append(f"step141_baseline_reference_status_invalid: {manifest.get('baseline_reference_status')}")
    if baseline_ids != EXPECTED_BASELINES:
        violations.append(f"step141_baseline_ids_mismatch: {sorted(baseline_ids)}")
    if int(manifest.get("ready_baseline_count", -1)) != 3:
        violations.append(f"step141_ready_baseline_count_not_3: {manifest.get('ready_baseline_count')}")
    if int(manifest.get("required_baseline_count", -1)) != 3:
        violations.append(f"step141_required_baseline_count_not_3: {manifest.get('required_baseline_count')}")
    if len(ready_refs) != 3:
        violations.append(f"step141_ready_ref_rows_not_3: {len(ready_refs)}")

    for ref in refs:
        bid = ref.get("baseline_id")
        if ref.get("reference_status") != "READY_AS_NONCAUSAL_BASELINE_REFERENCE":
            violations.append(f"{bid}_not_ready: {ref.get('reference_status')}")
        if bool(ref.get("causal_comparison_allowed", False)):
            violations.append(f"{bid}_causal_comparison_allowed_true")
        if bool(ref.get("paper_level_claim_allowed", False)):
            violations.append(f"{bid}_paper_level_claim_allowed_true")

    violations.extend([f"step141_manifest.{v}" for v in recursive_forbidden_true(manifest)])
    return result


def build_markdown(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 142 concrete release manifest draft validator gate",
        "",
        "This gate validates the Step 140 A-release draft and the Step 141 B-group baseline reference manifest.",
        "",
        "It does not execute reward ablation and does not release actual execution.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- gate_status: `{payload['gate_status']}`",
        f"- a_release_draft_validated: `{payload['a_release_draft_validated']}`",
        f"- b_group_baseline_ready: `{payload['b_group_baseline_ready']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Interpretation",
        "",
        "B0R/B1/B2 are ready as non-causal baseline references.",
        "The A-side concrete release draft is valid, but actual execution remains locked until a later explicit release process.",
        "",
    ])


def run_gate(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step140 = validate_step140(project_root)
    step141 = validate_step141(project_root)

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for component in [step140, step141]:
        warnings.extend([f"{component['component']}: {w}" for w in component.get("warnings", [])])
        blocking_reasons.extend([f"{component['component']}: {v}" for v in component.get("violations", [])])

    a_valid = len(step140.get("violations", [])) == 0
    b_ready = len(step141.get("violations", [])) == 0

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    gate_status = (
        "A_RELEASE_DRAFT_VALIDATED_AND_B_GROUP_BASELINES_READY_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "RELEASE_DRAFT_VALIDATOR_GATE_BLOCKED"
    )

    manifest_path = output_root / "concrete_release_manifest_draft_validator_gate_step142_manifest.json"
    md_path = output_root / "concrete_release_manifest_draft_validator_gate_step142.md"

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
        "gate_status": gate_status,
        "a_release_draft_validated": bool(a_valid),
        "b_group_baseline_ready": bool(b_ready),
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step140_check": step140,
        "step141_check": step141,
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
            "Step 142 validates Step 140 and Step 141 gates. "
            "It does not unlock actual reward ablation execution."
        ),
        "next_step_recommendation": (
            "Step 143 may expand the release matrix from A-only 18 runs to "
            "A/A90/A80/A70 72 runs before any explicit execution release."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "concrete_release_manifest_draft_validator_gate_step142.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "concrete_release_manifest_draft_validator_gate_step142.latest.json", {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "gate_status": gate_status,
        "a_release_draft_validated": bool(a_valid),
        "b_group_baseline_ready": bool(b_ready),
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/concrete_release_manifest_draft_validator_gate_step142")
    args = parser.parse_args()

    payload = run_gate(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 142 concrete release manifest draft validator gate completed")
    print(f"[OK] audit_status                : {payload['audit_status']}")
    print(f"[OK] gate_status                 : {payload['gate_status']}")
    print(f"[OK] a_release_draft_validated   : {payload['a_release_draft_validated']}")
    print(f"[OK] b_group_baseline_ready      : {payload['b_group_baseline_ready']}")
    print(f"[OK] actual_execution_allowed    : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released   : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed             : {payload['actual_executed']}")
    print(f"[OK] actual_results              : {payload['actual_results']}")
    print(f"[OK] winner_selected             : {payload['winner_selected']}")
    print(f"[OK] training_allowed            : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim                 : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest                    : {payload['manifest_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 142 validator gate failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
