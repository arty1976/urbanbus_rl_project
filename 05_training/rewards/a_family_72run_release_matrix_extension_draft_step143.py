from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 143
ARTIFACT_VERSION = "a_family_72run_release_matrix_extension_draft_step143_v1"

STEP142_POINTER = "05_training/rewards/concrete_release_manifest_draft_validator_gate_step142.latest.json"
STEP141_POINTER = "05_training/rewards/baseline_reference_manifest_step141.latest.json"
STEP140_POINTER = "05_training/rewards/concrete_reward_ablation_release_manifest_draft_step140.latest.json"

STEP142_FALLBACK_GLOB = "artifacts/rewards/**/*step142*manifest*.json"
STEP141_FALLBACK_GLOB = "artifacts/rewards/**/*step141*.json"
STEP140_FALLBACK_GLOB = "artifacts/rewards/**/*step140*manifest*.json"

CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = len(CONDITIONS) * len(REWARD_IDS) * len(SEEDS)

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
    fallback_glob: str,
    pointer_keys: List[str],
) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / pointer_rel

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            raw = None
            for key in pointer_keys:
                if payload.get(key):
                    raw = payload.get(key)
                    break
            if raw:
                p = resolve_path(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append(f"pointer_manifest_missing: {pointer_rel}")
            else:
                warnings.append(f"pointer_has_no_known_manifest_key: {pointer_rel}")
        except Exception as exc:
            warnings.append(f"pointer_unreadable: {pointer_rel}: {exc}")
    else:
        warnings.append(f"pointer_missing: {pointer_rel}")

    candidates = sorted(
        [p for p in project_root.glob(fallback_glob) if p.is_file() and "latest" not in p.name],
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


def load_required_manifest(
    project_root: Path,
    label: str,
    pointer_rel: str,
    fallback_glob: str,
) -> Dict[str, Any]:
    path, warnings = find_manifest(project_root, pointer_rel, fallback_glob, ["manifest_path"])
    result: Dict[str, Any] = {
        "label": label,
        "manifest_path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "warnings": warnings,
        "violations": [],
        "payload": {},
    }

    if path is None or not path.exists():
        result["violations"].append(f"{label}_manifest_missing")
        return result

    try:
        payload = load_json_any(path)
        result["payload"] = payload
    except Exception as exc:
        result["violations"].append(f"{label}_manifest_unreadable: {exc}")
        return result

    return result


def validate_step142_gate(project_root: Path) -> Dict[str, Any]:
    result = load_required_manifest(
        project_root,
        "step142",
        STEP142_POINTER,
        STEP142_FALLBACK_GLOB,
    )
    payload = result.get("payload", {})
    violations: List[str] = result["violations"]

    if not payload:
        return result

    result["summary"] = {
        "step": payload.get("step"),
        "audit_status": payload.get("audit_status"),
        "gate_status": payload.get("gate_status"),
        "a_release_draft_validated": bool(payload.get("a_release_draft_validated", False)),
        "b_group_baseline_ready": bool(payload.get("b_group_baseline_ready", False)),
    }

    if payload.get("step") != 142:
        violations.append(f"step142_step_mismatch: {payload.get('step')}")
    if payload.get("audit_status") != "PASS":
        violations.append(f"step142_audit_not_pass: {payload.get('audit_status')}")
    if payload.get("gate_status") != "A_RELEASE_DRAFT_VALIDATED_AND_B_GROUP_BASELINES_READY_ACTUAL_STILL_LOCKED":
        violations.append(f"step142_gate_status_invalid: {payload.get('gate_status')}")
    if not bool(payload.get("a_release_draft_validated", False)):
        violations.append("step142_a_release_draft_not_validated")
    if not bool(payload.get("b_group_baseline_ready", False)):
        violations.append("step142_b_group_baseline_not_ready")

    violations.extend([f"step142.{v}" for v in recursive_forbidden_true(payload)])
    return result


def validate_step141_baselines(project_root: Path) -> Dict[str, Any]:
    result = load_required_manifest(
        project_root,
        "step141",
        STEP141_POINTER,
        STEP141_FALLBACK_GLOB,
    )
    payload = result.get("payload", {})
    violations: List[str] = result["violations"]

    if not payload:
        return result

    refs = payload.get("baseline_references", [])
    baseline_ids = {str(r.get("baseline_id")) for r in refs}
    ready_ids = {
        str(r.get("baseline_id"))
        for r in refs
        if r.get("reference_status") == "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    }

    result["summary"] = {
        "step": payload.get("step"),
        "audit_status": payload.get("audit_status"),
        "baseline_reference_status": payload.get("baseline_reference_status"),
        "ready_baseline_count": payload.get("ready_baseline_count"),
        "required_baseline_count": payload.get("required_baseline_count"),
        "baseline_ids": sorted(baseline_ids),
        "ready_ids": sorted(ready_ids),
    }

    if payload.get("step") != 141:
        violations.append(f"step141_step_mismatch: {payload.get('step')}")
    if payload.get("audit_status") != "PASS":
        violations.append(f"step141_audit_not_pass: {payload.get('audit_status')}")
    if payload.get("baseline_reference_status") != "B_GROUP_BASELINE_REFERENCE_READY":
        violations.append(f"step141_reference_status_invalid: {payload.get('baseline_reference_status')}")
    if baseline_ids != EXPECTED_BASELINES:
        violations.append(f"step141_baseline_ids_mismatch: {sorted(baseline_ids)}")
    if ready_ids != EXPECTED_BASELINES:
        violations.append(f"step141_ready_ids_mismatch: {sorted(ready_ids)}")

    violations.extend([f"step141.{v}" for v in recursive_forbidden_true(payload)])
    return result


def validate_step140_source(project_root: Path) -> Dict[str, Any]:
    result = load_required_manifest(
        project_root,
        "step140",
        STEP140_POINTER,
        STEP140_FALLBACK_GLOB,
    )
    payload = result.get("payload", {})
    violations: List[str] = result["violations"]

    if not payload:
        return result

    result["summary"] = {
        "step": payload.get("step"),
        "audit_status": payload.get("audit_status"),
        "draft_status": payload.get("draft_status"),
        "release_validation_status": payload.get("release_validation_status"),
        "concrete_draft_path": payload.get("concrete_draft_path"),
    }

    if payload.get("step") != 140:
        violations.append(f"step140_step_mismatch: {payload.get('step')}")
    if payload.get("audit_status") != "PASS":
        violations.append(f"step140_audit_not_pass: {payload.get('audit_status')}")
    if payload.get("draft_status") != "CONCRETE_RELEASE_MANIFEST_DRAFT_CREATED_ACTUAL_STILL_LOCKED":
        violations.append(f"step140_draft_status_invalid: {payload.get('draft_status')}")

    violations.extend([f"step140.{v}" for v in recursive_forbidden_true(payload)])
    return result


def build_run_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for condition in CONDITIONS:
        for reward_id in REWARD_IDS:
            for seed in SEEDS:
                ordinal += 1
                rows.append({
                    "run_ordinal": ordinal,
                    "run_id": f"{condition}_{reward_id}_seed_{seed:03d}",
                    "condition_id": condition,
                    "reward_id": reward_id,
                    "seed": seed,
                    "qwen_train": False,
                    "qwen_inference": False,
                    "actual_execution_allowed": False,
                    "actual_execution_released": False,
                    "expected_output_mode": "canonical_kpi_after_actual_ablation",
                    "baseline_reference_group": "B0R_B1_B2",
                })
    return rows


def validate_matrix(matrix: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    if len(matrix) != EXPECTED_RUN_COUNT:
        errors.append(f"matrix_run_count_mismatch: expected={EXPECTED_RUN_COUNT} got={len(matrix)}")

    run_ids = [str(r.get("run_id")) for r in matrix]
    if len(set(run_ids)) != len(run_ids):
        errors.append("matrix_run_id_duplicate")

    condition_counts = Counter(str(r.get("condition_id")) for r in matrix)
    reward_counts = Counter(str(r.get("reward_id")) for r in matrix)
    seed_counts = Counter(int(r.get("seed")) for r in matrix)

    for condition in CONDITIONS:
        if condition_counts[condition] != len(REWARD_IDS) * len(SEEDS):
            errors.append(f"condition_count_mismatch: {condition}={condition_counts[condition]}")

    for reward_id in REWARD_IDS:
        if reward_counts[reward_id] != len(CONDITIONS) * len(SEEDS):
            errors.append(f"reward_count_mismatch: {reward_id}={reward_counts[reward_id]}")

    for seed in SEEDS:
        if seed_counts[seed] != len(CONDITIONS) * len(REWARD_IDS):
            errors.append(f"seed_count_mismatch: {seed}={seed_counts[seed]}")

    for row in matrix:
        if row.get("condition_id") not in CONDITIONS:
            errors.append(f"unexpected_condition: {row.get('condition_id')}")
        if row.get("reward_id") not in REWARD_IDS:
            errors.append(f"unexpected_reward_id: {row.get('reward_id')}")
        if int(row.get("seed")) not in SEEDS:
            errors.append(f"unexpected_seed: {row.get('seed')}")
        if bool(row.get("qwen_train", True)):
            errors.append(f"{row.get('run_id')}: qwen_train_must_be_false")
        if bool(row.get("qwen_inference", True)):
            errors.append(f"{row.get('run_id')}: qwen_inference_must_be_false")
        if bool(row.get("actual_execution_allowed", False)):
            errors.append(f"{row.get('run_id')}: actual_execution_allowed_true")
        if bool(row.get("actual_execution_released", False)):
            errors.append(f"{row.get('run_id')}: actual_execution_released_true")

    return errors


def build_markdown(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 143 A-family 72-run release matrix extension draft",
        "",
        "This step expands the A-only Step 140 release draft into an A-family matrix draft.",
        "",
        "It does not execute reward ablation and does not release actual execution.",
        "",
        "## Matrix",
        "",
        "- conditions: `A`, `A90`, `A80`, `A70`",
        "- reward_ids: `R0` through `R5`",
        "- seeds: `1`, `2`, `3`",
        "- planned_run_count: `72`",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- matrix_status: `{payload['matrix_status']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Comparison plan",
        "",
        "After future H200 execution, A-family outputs should be aggregated to canonical KPI schema and compared against Step 141 B0R/B1/B2 baseline references.",
        "",
    ])


def run_generator(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step142 = validate_step142_gate(project_root)
    step141 = validate_step141_baselines(project_root)
    step140 = validate_step140_source(project_root)
    run_matrix = build_run_matrix()

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for check in [step142, step141, step140]:
        warnings.extend([f"{check['label']}: {w}" for w in check.get("warnings", [])])
        blocking_reasons.extend([f"{check['label']}: {v}" for v in check.get("violations", [])])

    blocking_reasons.extend(validate_matrix(run_matrix))

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    matrix_status = (
        "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_READY_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_BLOCKED"
    )

    condition_counts = dict(Counter(str(r["condition_id"]) for r in run_matrix))
    reward_counts = dict(Counter(str(r["reward_id"]) for r in run_matrix))
    seed_counts = {str(k): v for k, v in Counter(int(r["seed"]) for r in run_matrix).items()}

    draft_path = output_root / "a_family_72run_release_matrix_extension_draft_step143.json"
    manifest_path = output_root / "a_family_72run_release_matrix_extension_draft_step143_manifest.json"
    md_path = output_root / "a_family_72run_release_matrix_extension_draft_step143.md"

    draft = {
        "release_matrix_version": "a_family_72run_release_matrix_v1",
        "created_from_step": 143,
        "purpose": "Draft only. This file does not release actual execution.",
        "conditions": CONDITIONS,
        "reward_ids": REWARD_IDS,
        "seeds": SEEDS,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "run_matrix": run_matrix,
        "condition_counts": condition_counts,
        "reward_counts": reward_counts,
        "seed_counts": seed_counts,
        "source_steps": {
            "step140_a_only_release_draft": step140.get("manifest_path", ""),
            "step141_b_group_baseline_reference": step141.get("manifest_path", ""),
            "step142_validator_gate": step142.get("manifest_path", ""),
        },
        "baseline_reference": {
            "baseline_group": ["B0R", "B1", "B2"],
            "baseline_reference_manifest": step141.get("manifest_path", ""),
            "comparison_after_execution": True,
            "comparison_allowed_now": False,
        },
        "release_decision": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "release_ready": False,
            "release_blocking_reason": "step143_is_matrix_extension_draft_not_operator_release",
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
        "matrix_status": matrix_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "conditions": CONDITIONS,
        "reward_ids": REWARD_IDS,
        "seeds": SEEDS,
        "condition_counts": condition_counts,
        "reward_counts": reward_counts,
        "seed_counts": seed_counts,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "step140_check": step140,
        "step141_check": step141,
        "step142_check": step142,
        "draft_path": str(draft_path),
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
            "Step 143 only drafts the A-family 72-run release matrix. "
            "It does not run H200 jobs and does not unlock actual execution."
        ),
        "next_step_recommendation": (
            "Step 144 should validate whether this 72-run matrix is the final H200 matrix "
            "or whether execution should remain staged by condition."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(draft_path, draft)
    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "a_family_72run_release_matrix_extension_draft_step143.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "a_family_72run_release_matrix_extension_draft_step143.latest.json", {
        "manifest_path": str(manifest_path),
        "draft_path": str(draft_path),
        "audit_status": audit_status,
        "matrix_status": matrix_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/a_family_72run_release_matrix_extension_draft_step143")
    args = parser.parse_args()

    payload = run_generator(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 143 A-family 72-run release matrix extension draft completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] matrix_status             : {payload['matrix_status']}")
    print(f"[OK] planned_run_count         : {payload['planned_run_count']}")
    print(f"[OK] conditions                : {payload['conditions']}")
    print(f"[OK] reward_ids                : {payload['reward_ids']}")
    print(f"[OK] seeds                     : {payload['seeds']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] draft                     : {payload['draft_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 143 matrix extension draft failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
