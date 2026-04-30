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

$PyMain = Join-Path $RewardsDir "h200_transfer_package_export_manifest_step145.py"
$PyValidate = Join-Path $RewardsDir "validate_h200_transfer_package_export_manifest_step145.py"
$PyTest = Join-Path $RewardsDir "test_h200_transfer_package_export_manifest_step145.py"
$MdDoc = Join-Path $RewardsDir "h200_transfer_package_export_manifest_step145.md"

$MainCode = @'
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 145
ARTIFACT_VERSION = "h200_transfer_package_export_manifest_step145_v1"

STEP144_POINTER = "05_training/rewards/h200_execution_package_boundary_manifest_step144.latest.json"
STEP144_FALLBACK_MANIFEST = "artifacts/rewards/h200_execution_package_boundary_manifest_step144/h200_execution_package_boundary_manifest_step144.json"
STEP144_FALLBACK_BOUNDARY = "artifacts/rewards/h200_execution_package_boundary_manifest_step144/h200_execution_package_boundary_step144.json"

EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

FORBIDDEN_TRANSFER_PREFIXES = [
    "artifacts/",
    "artifacts\\",
    "1000005000/",
    "1000005000\\",
    "project_files/",
    "project_files\\",
]

FORBIDDEN_TRANSFER_SUFFIXES = [
    ".latest.json",
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


def to_posix_rel(project_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return rel.as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def git_status_short(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def find_step144(project_root: Path) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP144_POINTER

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            manifest_raw = payload.get("manifest_path")
            boundary_raw = payload.get("boundary_path")
            manifest_path = resolve_path(project_root, str(manifest_raw)) if manifest_raw else None
            boundary_path = resolve_path(project_root, str(boundary_raw)) if boundary_raw else None

            if manifest_path and manifest_path.exists() and boundary_path and boundary_path.exists():
                return manifest_path, boundary_path, warnings

            if manifest_path and not manifest_path.exists():
                warnings.append(f"step144_pointer_manifest_missing: {manifest_path}")
            if boundary_path and not boundary_path.exists():
                warnings.append(f"step144_pointer_boundary_missing: {boundary_path}")
            if not manifest_raw:
                warnings.append("step144_pointer_has_no_manifest_path")
            if not boundary_raw:
                warnings.append("step144_pointer_has_no_boundary_path")
        except Exception as exc:
            warnings.append(f"step144_pointer_unreadable: {exc}")
    else:
        warnings.append(f"step144_pointer_missing: {STEP144_POINTER}")

    fallback_manifest = project_root / STEP144_FALLBACK_MANIFEST
    fallback_boundary = project_root / STEP144_FALLBACK_BOUNDARY
    if fallback_manifest.exists() and fallback_boundary.exists():
        warnings.append("using_step144_fallback_manifest_and_boundary")
        return fallback_manifest, fallback_boundary, warnings

    return None, None, warnings


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


def is_forbidden_transfer_path(rel_path: str) -> List[str]:
    normalized = rel_path.replace("\\", "/")
    violations: List[str] = []
    for prefix in FORBIDDEN_TRANSFER_PREFIXES:
        if normalized.startswith(prefix.replace("\\", "/")):
            violations.append(f"forbidden_transfer_prefix: {prefix}")
    for suffix in FORBIDDEN_TRANSFER_SUFFIXES:
        if normalized.endswith(suffix):
            violations.append(f"forbidden_transfer_suffix: {suffix}")
    return violations


def validate_step144_payloads(
    manifest_path: Optional[Path],
    boundary_path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    violations: List[str] = []
    manifest: Dict[str, Any] = {}
    boundary: Dict[str, Any] = {}

    if manifest_path is None or not manifest_path.exists():
        violations.append("step144_manifest_missing")
        return manifest, boundary, violations

    if boundary_path is None or not boundary_path.exists():
        violations.append("step144_boundary_missing")
        return manifest, boundary, violations

    try:
        manifest = load_json_any(manifest_path)
    except Exception as exc:
        violations.append(f"step144_manifest_unreadable: {exc}")
        return manifest, boundary, violations

    try:
        boundary = load_json_any(boundary_path)
    except Exception as exc:
        violations.append(f"step144_boundary_unreadable: {exc}")
        return manifest, boundary, violations

    if manifest.get("step") != 144:
        violations.append(f"step144_step_mismatch: {manifest.get('step')}")
    if manifest.get("artifact_version") != "h200_execution_package_boundary_manifest_step144_v1":
        violations.append(f"step144_artifact_version_mismatch: {manifest.get('artifact_version')}")
    if manifest.get("audit_status") != "PASS":
        violations.append(f"step144_audit_not_pass: {manifest.get('audit_status')}")
    if manifest.get("package_status") != "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED":
        violations.append(f"step144_package_status_invalid: {manifest.get('package_status')}")
    if int(manifest.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        violations.append(f"step144_planned_run_count_not_72: {manifest.get('planned_run_count')}")
    if manifest.get("conditions") != EXPECTED_CONDITIONS:
        violations.append(f"step144_conditions_mismatch: {manifest.get('conditions')}")
    if manifest.get("reward_ids") != EXPECTED_REWARD_IDS:
        violations.append(f"step144_reward_ids_mismatch: {manifest.get('reward_ids')}")
    if manifest.get("seeds") != EXPECTED_SEEDS:
        violations.append(f"step144_seeds_mismatch: {manifest.get('seeds')}")

    if boundary.get("package_boundary_version") != "h200_execution_package_boundary_v1":
        violations.append(f"boundary_version_mismatch: {boundary.get('package_boundary_version')}")
    if boundary.get("created_from_step") != 144:
        violations.append(f"boundary_created_from_step_mismatch: {boundary.get('created_from_step')}")

    matrix = boundary.get("matrix_source", {})
    if int(matrix.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        violations.append(f"boundary_matrix_run_count_not_72: {matrix.get('planned_run_count')}")
    if matrix.get("conditions") != EXPECTED_CONDITIONS:
        violations.append(f"boundary_matrix_conditions_mismatch: {matrix.get('conditions')}")
    if matrix.get("reward_ids") != EXPECTED_REWARD_IDS:
        violations.append(f"boundary_matrix_reward_ids_mismatch: {matrix.get('reward_ids')}")
    if matrix.get("seeds") != EXPECTED_SEEDS:
        violations.append(f"boundary_matrix_seeds_mismatch: {matrix.get('seeds')}")

    violations.extend([f"step144_manifest.{v}" for v in recursive_forbidden_true(manifest)])
    violations.extend([f"step144_boundary.{v}" for v in recursive_forbidden_true(boundary)])

    return manifest, boundary, violations


def build_transfer_entries(project_root: Path, boundary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    warnings: List[str] = []
    violations: List[str] = []

    raw_candidates = boundary.get("h200_transfer_candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        violations.append("h200_transfer_candidates_missing_or_empty")
        return [], warnings, violations

    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for item in raw_candidates:
        raw_path = str(item.get("path", ""))
        if not raw_path:
            violations.append("transfer_candidate_empty_path")
            continue

        p = resolve_path(project_root, raw_path)
        rel_path = to_posix_rel(project_root, p)

        if rel_path in seen:
            violations.append(f"duplicate_transfer_candidate: {rel_path}")
            continue
        seen.add(rel_path)

        forbid = is_forbidden_transfer_path(rel_path)
        for f in forbid:
            violations.append(f"{rel_path}: {f}")

        exists = p.exists() and p.is_file()
        entry: Dict[str, Any] = {
            "relative_path": rel_path,
            "absolute_path": str(p),
            "exists": bool(exists),
            "size_bytes": int(p.stat().st_size) if exists else 0,
            "sha256": sha256_file(p) if exists else "",
            "role": item.get("role", "source_or_contract_for_h200_transfer"),
            "transfer_required": True,
            "h200_target_path": f"urbanbus_rl_project/{rel_path}",
        }

        if not exists:
            violations.append(f"transfer_file_missing: {rel_path}")

        entries.append(entry)

    return entries, warnings, violations


def build_filelist(entries: List[Dict[str, Any]]) -> str:
    lines = [
        "# Step 145 H200 transfer package file list",
        "# Format: sha256  size_bytes  relative_path",
    ]
    for entry in entries:
        lines.append(
            f"{entry.get('sha256', '')}  {entry.get('size_bytes', 0)}  {entry.get('relative_path', '')}"
        )
    return "\n".join(lines) + "\n"


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Step 145 H200 transfer package export manifest",
        "",
        "This step exports the file-level manifest for the H200 transfer package.",
        "",
        "It does not run H200 jobs and does not release actual execution.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- export_status: `{payload['export_status']}`",
        f"- transfer_file_count: `{payload['transfer_file_count']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Transfer files",
        "",
        "| # | relative_path | size_bytes | sha256 |",
        "|---:|---|---:|---|",
    ]
    for idx, entry in enumerate(payload.get("transfer_entries", []), start=1):
        sha = str(entry.get("sha256", ""))
        lines.append(
            f"| {idx} | `{entry.get('relative_path', '')}` | {entry.get('size_bytes', 0)} | `{sha[:12]}...` |"
        )

    lines.extend([
        "",
        "## Guard",
        "",
        "- `artifacts/**`, `*.latest.json`, baseline artifact folders, API sampling folders, and temporary project_files are not transfer source-of-truth targets.",
        "- Step 143 remains the 72-run matrix source of truth for H200 planning.",
        "- Actual execution remains locked until a later explicit operator release.",
        "",
    ])

    return "\n".join(lines)


def run_export(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    warnings: List[str] = []
    blocking_reasons: List[str] = []

    step144_manifest_path, step144_boundary_path, find_warnings = find_step144(project_root)
    warnings.extend(find_warnings)

    step144_manifest, step144_boundary, step144_violations = validate_step144_payloads(
        step144_manifest_path,
        step144_boundary_path,
    )
    blocking_reasons.extend(step144_violations)

    transfer_entries: List[Dict[str, Any]] = []
    if step144_boundary:
        entries, entry_warnings, entry_violations = build_transfer_entries(project_root, step144_boundary)
        transfer_entries = entries
        warnings.extend(entry_warnings)
        blocking_reasons.extend(entry_violations)

    git_hash = git_commit_hash(project_root)
    git_dirty = git_status_short(project_root)

    # A dirty tree is allowed here because Step 145 files are created by this script.
    # Actual H200 execution still requires a clean or recorded dirty state.
    if git_dirty:
        warnings.append("git_status_not_clean_at_export_time_step145_files_may_be_uncommitted")

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    export_status = (
        "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_BLOCKED"
    )

    export_manifest_path = output_root / "h200_transfer_package_export_manifest_step145.json"
    filelist_path = output_root / "h200_transfer_package_filelist_step145.txt"
    md_path = output_root / "h200_transfer_package_export_manifest_step145.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_hash,
        "git_status_clean": not bool(git_dirty),
        "git_status_short": git_dirty,
        "audit_status": audit_status,
        "export_status": export_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "conditions": EXPECTED_CONDITIONS,
        "reward_ids": EXPECTED_REWARD_IDS,
        "seeds": EXPECTED_SEEDS,
        "step144_manifest_path": str(step144_manifest_path) if step144_manifest_path else "",
        "step144_boundary_path": str(step144_boundary_path) if step144_boundary_path else "",
        "transfer_file_count": len(transfer_entries),
        "transfer_entries": transfer_entries,
        "forbidden_transfer_prefixes": FORBIDDEN_TRANSFER_PREFIXES,
        "forbidden_transfer_suffixes": FORBIDDEN_TRANSFER_SUFFIXES,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
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
            "Step 145 exports a H200 transfer package manifest only. "
            "It does not copy files to H200 and does not unlock execution."
        ),
        "next_step_recommendation": (
            "Step 146 should create a transfer package integrity verifier to run after files are copied to H200."
        ),
        "export_manifest_path": str(export_manifest_path),
        "filelist_path": str(filelist_path),
        "markdown_path": str(md_path),
    }

    dump_json(export_manifest_path, payload)
    dump_text(filelist_path, build_filelist(transfer_entries))
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "h200_transfer_package_export_manifest_step145.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "h200_transfer_package_export_manifest_step145.latest.json", {
        "manifest_path": str(export_manifest_path),
        "filelist_path": str(filelist_path),
        "audit_status": audit_status,
        "export_status": export_status,
        "transfer_file_count": len(transfer_entries),
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
    parser.add_argument("--output-root", default="artifacts/rewards/h200_transfer_package_export_manifest_step145")
    args = parser.parse_args()

    payload = run_export(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 145 H200 transfer package export manifest completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] export_status             : {payload['export_status']}")
    print(f"[OK] transfer_file_count       : {payload['transfer_file_count']}")
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
    print(f"[OK] filelist                  : {payload['filelist_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 145 transfer export manifest failed")
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


ARTIFACT_VERSION = "h200_transfer_package_export_manifest_step145_v1"
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

FORBIDDEN_TRANSFER_PREFIXES = [
    "artifacts/",
    "1000005000/",
    "project_files/",
]
FORBIDDEN_TRANSFER_SUFFIXES = [
    ".latest.json",
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


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                errors.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    return errors


def validate_transfer_entries(entries: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    if not entries:
        errors.append("transfer_entries_empty")

    seen: set[str] = set()
    for entry in entries:
        rel = str(entry.get("relative_path", "")).replace("\\", "/")
        if not rel:
            errors.append("transfer_entry_empty_relative_path")
            continue

        if rel in seen:
            errors.append(f"duplicate_transfer_entry: {rel}")
        seen.add(rel)

        for prefix in FORBIDDEN_TRANSFER_PREFIXES:
            if rel.startswith(prefix):
                errors.append(f"{rel}: forbidden_transfer_prefix_{prefix}")
        for suffix in FORBIDDEN_TRANSFER_SUFFIXES:
            if rel.endswith(suffix):
                errors.append(f"{rel}: forbidden_transfer_suffix_{suffix}")

        if not bool(entry.get("exists", False)):
            errors.append(f"{rel}: transfer_entry_missing")
        if int(entry.get("size_bytes", 0)) <= 0:
            errors.append(f"{rel}: transfer_entry_empty_file")
        sha = str(entry.get("sha256", ""))
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            errors.append(f"{rel}: invalid_sha256")

    return errors


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 145:
        errors.append("step_must_be_145")

    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    if payload.get("export_status") not in {
        "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED",
        "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_BLOCKED",
    }:
        errors.append("invalid_export_status")
    if require_pass and payload.get("export_status") != "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED":
        errors.append("export_status_not_ready_locked")

    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("planned_run_count_must_be_72")
    if payload.get("conditions") != EXPECTED_CONDITIONS:
        errors.append("conditions_mismatch")
    if payload.get("reward_ids") != EXPECTED_REWARD_IDS:
        errors.append("reward_ids_mismatch")
    if payload.get("seeds") != EXPECTED_SEEDS:
        errors.append("seeds_mismatch")

    entries = payload.get("transfer_entries", [])
    if int(payload.get("transfer_file_count", -1)) != len(entries):
        errors.append("transfer_file_count_mismatch")

    errors.extend(validate_transfer_entries(entries))
    errors.extend(recursive_forbidden_true(payload))

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
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 145 H200 transfer package export manifest validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

from validate_h200_transfer_package_export_manifest_step145 import validate_payload


def base_payload() -> dict:
    entries = [
        {
            "relative_path": "05_training/rewards/final_reward_spec_step111.md",
            "exists": True,
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
        {
            "relative_path": "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.py",
            "exists": True,
            "size_bytes": 10,
            "sha256": "b" * 64,
        },
    ]
    return {
        "artifact_version": "h200_transfer_package_export_manifest_step145_v1",
        "step": 145,
        "audit_status": "PASS",
        "export_status": "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED",
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "transfer_file_count": len(entries),
        "transfer_entries": entries,
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
    }


def test_pass_payload() -> None:
    result = validate_payload(base_payload(), require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_actual_execution_true_fails() -> None:
    payload = base_payload()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_forbidden_artifact_transfer_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["relative_path"] = "artifacts/rewards/foo.json"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("forbidden_transfer_prefix" in e for e in result["errors"])


def test_latest_pointer_transfer_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["relative_path"] = "05_training/rewards/foo.latest.json"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("forbidden_transfer_suffix" in e for e in result["errors"])


def test_bad_sha_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["sha256"] = "not-a-sha"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("invalid_sha256" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["export_status"] = "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_true_fails()
    test_forbidden_artifact_transfer_fails()
    test_latest_pointer_transfer_fails()
    test_bad_sha_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 145 H200 transfer package export manifest self-test PASS")


if __name__ == "__main__":
    main()
'@

$DocText = @'
# Step 145 H200 transfer package export manifest

Step 145 creates the H200 transfer package export manifest.

It reads Step 144 boundary output and records:
- transfer candidate file list
- file existence
- file size
- SHA256 hash
- H200 target path
- forbidden local-only path checks

This step does not copy files to H200.
This step does not execute H200 jobs.
This step does not release actual execution.
This step does not select a reward winner.
This step does not allow paper-level claims.

The H200 transfer package source of truth remains tracked source/spec/validator/test files, not artifacts or latest pointer files.
'@

Set-Content -Path $PyMain -Value $MainCode -Encoding UTF8
Set-Content -Path $PyValidate -Value $ValidateCode -Encoding UTF8
Set-Content -Path $PyTest -Value $TestCode -Encoding UTF8
Set-Content -Path $MdDoc -Value $DocText -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = "artifacts/rewards/h200_transfer_package_export_manifest_step145"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\h200_transfer_package_export_manifest_step145.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 145 H200 transfer package export manifest failed"
}

& $py $PyValidate --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 145 transfer export manifest validation failed"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 145 self-test failed"
}

Write-Host "[DONE] Step 145 H200 transfer package export manifest complete."
