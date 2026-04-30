from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 146
ARTIFACT_VERSION = "h200_transfer_package_integrity_verifier_step146_v1"

STEP145_POINTER = "05_training/rewards/h200_transfer_package_export_manifest_step145.latest.json"
STEP145_FALLBACK_MANIFEST = "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_export_manifest_step145.json"
STEP145_FALLBACK_FILELIST = "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_filelist_step145.txt"

EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

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


def resolve_path(base_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = base_root / p
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def find_step145(project_root: Path) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP145_POINTER

    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            manifest_raw = payload.get("manifest_path")
            filelist_raw = payload.get("filelist_path")
            manifest_path = resolve_path(project_root, str(manifest_raw)) if manifest_raw else None
            filelist_path = resolve_path(project_root, str(filelist_raw)) if filelist_raw else None

            if manifest_path and manifest_path.exists() and filelist_path and filelist_path.exists():
                return manifest_path, filelist_path, warnings

            if manifest_path and not manifest_path.exists():
                warnings.append(f"step145_pointer_manifest_missing: {manifest_path}")
            if filelist_path and not filelist_path.exists():
                warnings.append(f"step145_pointer_filelist_missing: {filelist_path}")
            if not manifest_raw:
                warnings.append("step145_pointer_has_no_manifest_path")
            if not filelist_raw:
                warnings.append("step145_pointer_has_no_filelist_path")
        except Exception as exc:
            warnings.append(f"step145_pointer_unreadable: {exc}")
    else:
        warnings.append(f"step145_pointer_missing: {STEP145_POINTER}")

    fallback_manifest = project_root / STEP145_FALLBACK_MANIFEST
    fallback_filelist = project_root / STEP145_FALLBACK_FILELIST
    if fallback_manifest.exists() and fallback_filelist.exists():
        warnings.append("using_step145_fallback_manifest_and_filelist")
        return fallback_manifest, fallback_filelist, warnings

    return None, None, warnings


def validate_step145_manifest(path: Optional[Path]) -> Tuple[Dict[str, Any], List[str]]:
    violations: List[str] = []
    payload: Dict[str, Any] = {}

    if path is None or not path.exists():
        violations.append("step145_manifest_missing")
        return payload, violations

    try:
        payload = load_json_any(path)
    except Exception as exc:
        violations.append(f"step145_manifest_unreadable: {exc}")
        return payload, violations

    if payload.get("step") != 145:
        violations.append(f"step145_step_mismatch: {payload.get('step')}")
    if payload.get("artifact_version") != "h200_transfer_package_export_manifest_step145_v1":
        violations.append(f"step145_artifact_version_mismatch: {payload.get('artifact_version')}")
    if payload.get("audit_status") != "PASS":
        violations.append(f"step145_audit_not_pass: {payload.get('audit_status')}")
    if payload.get("export_status") != "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED":
        violations.append(f"step145_export_status_invalid: {payload.get('export_status')}")
    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        violations.append(f"step145_planned_run_count_not_72: {payload.get('planned_run_count')}")
    if payload.get("conditions") != EXPECTED_CONDITIONS:
        violations.append(f"step145_conditions_mismatch: {payload.get('conditions')}")
    if payload.get("reward_ids") != EXPECTED_REWARD_IDS:
        violations.append(f"step145_reward_ids_mismatch: {payload.get('reward_ids')}")
    if payload.get("seeds") != EXPECTED_SEEDS:
        violations.append(f"step145_seeds_mismatch: {payload.get('seeds')}")

    entries = payload.get("transfer_entries", [])
    if not isinstance(entries, list) or not entries:
        violations.append("step145_transfer_entries_missing")
    if int(payload.get("transfer_file_count", -1)) != len(entries):
        violations.append("step145_transfer_file_count_mismatch")

    violations.extend([f"step145_manifest.{v}" for v in recursive_forbidden_true(payload)])
    return payload, violations


def parse_filelist(filelist_path: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
    violations: List[str] = []
    rows: List[Dict[str, Any]] = []

    if filelist_path is None or not filelist_path.exists():
        violations.append("step145_filelist_missing")
        return rows, violations

    try:
        text = filelist_path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        violations.append(f"step145_filelist_unreadable: {exc}")
        return rows, violations

    for lineno, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        parts = raw.split(maxsplit=2)
        if len(parts) != 3:
            violations.append(f"filelist_line_invalid: line={lineno}")
            continue

        sha, size_raw, rel = parts
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            violations.append(f"filelist_sha_invalid: line={lineno}")
        try:
            size = int(size_raw)
        except Exception:
            size = -1
            violations.append(f"filelist_size_invalid: line={lineno}")

        rows.append({
            "sha256": sha.lower(),
            "size_bytes": size,
            "relative_path": rel.replace("\\", "/"),
            "line_no": lineno,
        })

    if not rows:
        violations.append("filelist_has_no_entries")

    return rows, violations


def verify_entries(transfer_root: Path, expected_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    violations: List[str] = []
    verified: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for row in expected_rows:
        rel = str(row.get("relative_path", "")).replace("\\", "/")
        if rel in seen:
            violations.append(f"duplicate_expected_file: {rel}")
        seen.add(rel)

        expected_size = int(row.get("size_bytes", -1))
        expected_sha = str(row.get("sha256", "")).lower()

        path = resolve_path(transfer_root, rel)
        exists = path.exists() and path.is_file()
        actual_size = int(path.stat().st_size) if exists else 0
        actual_sha = sha256_file(path) if exists else ""

        status = "PASS"
        reasons: List[str] = []

        if not exists:
            status = "FAIL"
            reasons.append("missing")
        else:
            if actual_size != expected_size:
                status = "FAIL"
                reasons.append(f"size_mismatch: expected={expected_size} actual={actual_size}")
            if actual_sha.lower() != expected_sha:
                status = "FAIL"
                reasons.append("sha256_mismatch")

        if status != "PASS":
            violations.append(f"{rel}: {'; '.join(reasons)}")

        verified.append({
            "relative_path": rel,
            "checked_path": str(path),
            "exists": bool(exists),
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "integrity_status": status,
            "reasons": reasons,
        })

    return verified, violations


def cross_check_manifest_entries(manifest: Dict[str, Any], filelist_rows: List[Dict[str, Any]]) -> List[str]:
    violations: List[str] = []
    manifest_entries = manifest.get("transfer_entries", [])

    manifest_map = {
        str(e.get("relative_path", "")).replace("\\", "/"): {
            "sha256": str(e.get("sha256", "")).lower(),
            "size_bytes": int(e.get("size_bytes", -1)),
        }
        for e in manifest_entries
    }
    filelist_map = {
        str(r.get("relative_path", "")).replace("\\", "/"): {
            "sha256": str(r.get("sha256", "")).lower(),
            "size_bytes": int(r.get("size_bytes", -1)),
        }
        for r in filelist_rows
    }

    if set(manifest_map) != set(filelist_map):
        violations.append(
            f"manifest_filelist_paths_mismatch: manifest_only={sorted(set(manifest_map)-set(filelist_map))} "
            f"filelist_only={sorted(set(filelist_map)-set(manifest_map))}"
        )

    for rel in sorted(set(manifest_map).intersection(filelist_map)):
        if manifest_map[rel]["sha256"] != filelist_map[rel]["sha256"]:
            violations.append(f"{rel}: manifest_filelist_sha_mismatch")
        if manifest_map[rel]["size_bytes"] != filelist_map[rel]["size_bytes"]:
            violations.append(f"{rel}: manifest_filelist_size_mismatch")

    return violations


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Step 146 H200 transfer package integrity verifier",
        "",
        "This step verifies copied H200 transfer package files against the Step 145 filelist and SHA256 hashes.",
        "",
        "It does not run H200 jobs and does not release actual execution.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- integrity_status: `{payload['integrity_status']}`",
        f"- verified_file_count: `{payload['verified_file_count']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Verified files",
        "",
        "| # | relative_path | status | size | sha256 |",
        "|---:|---|---|---:|---|",
    ]
    for idx, entry in enumerate(payload.get("verified_entries", []), start=1):
        sha = str(entry.get("actual_sha256") or entry.get("expected_sha256") or "")
        lines.append(
            f"| {idx} | `{entry.get('relative_path', '')}` | `{entry.get('integrity_status', '')}` | {entry.get('actual_size_bytes', 0)} | `{sha[:12]}...` |"
        )

    lines.extend([
        "",
        "## Guard",
        "",
        "- Integrity PASS only means the transfer package files match the Step 145 filelist.",
        "- It does not permit actual execution, winner selection, training claim, or paper-level claim.",
        "",
    ])

    return "\n".join(lines)


def run_verifier(project_root: Path, transfer_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    transfer_root = transfer_root.resolve()
    output_root = output_root.resolve()

    warnings: List[str] = []
    blocking_reasons: List[str] = []

    step145_manifest_path, step145_filelist_path, find_warnings = find_step145(project_root)
    warnings.extend(find_warnings)

    step145_manifest, step145_violations = validate_step145_manifest(step145_manifest_path)
    blocking_reasons.extend(step145_violations)

    filelist_rows, filelist_violations = parse_filelist(step145_filelist_path)
    blocking_reasons.extend(filelist_violations)

    if step145_manifest and filelist_rows:
        blocking_reasons.extend(cross_check_manifest_entries(step145_manifest, filelist_rows))

    verified_entries: List[Dict[str, Any]] = []
    if filelist_rows:
        verified_entries, verify_violations = verify_entries(transfer_root, filelist_rows)
        blocking_reasons.extend(verify_violations)

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    integrity_status = (
        "H200_TRANSFER_PACKAGE_INTEGRITY_VERIFIED_ACTUAL_STILL_LOCKED"
        if audit_status == "PASS"
        else "H200_TRANSFER_PACKAGE_INTEGRITY_BLOCKED"
    )

    manifest_path = output_root / "h200_transfer_package_integrity_verifier_step146_manifest.json"
    md_path = output_root / "h200_transfer_package_integrity_verifier_step146.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "transfer_root": str(transfer_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "integrity_status": integrity_status,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "conditions": EXPECTED_CONDITIONS,
        "reward_ids": EXPECTED_REWARD_IDS,
        "seeds": EXPECTED_SEEDS,
        "step145_manifest_path": str(step145_manifest_path) if step145_manifest_path else "",
        "step145_filelist_path": str(step145_filelist_path) if step145_filelist_path else "",
        "verified_file_count": len(verified_entries),
        "verified_entries": verified_entries,
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
            "Step 146 verifies transfer package file integrity only. "
            "It does not execute H200 jobs and does not unlock actual execution."
        ),
        "next_step_recommendation": (
            "Step 147 should create the H200-side receive/runbook checklist or rsync/copy command plan."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "h200_transfer_package_integrity_verifier_step146.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "h200_transfer_package_integrity_verifier_step146.latest.json", {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "integrity_status": integrity_status,
        "verified_file_count": len(verified_entries),
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
    parser.add_argument(
        "--transfer-root",
        default=".",
        help="Root of copied package on H200. For local self-check, keep '.'.",
    )
    parser.add_argument("--output-root", default="artifacts/rewards/h200_transfer_package_integrity_verifier_step146")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    transfer_root = resolve_path(project_root, args.transfer_root)
    output_root = project_root / args.output_root

    payload = run_verifier(project_root, transfer_root, output_root)

    print("[OK] Step 146 H200 transfer package integrity verifier completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] integrity_status          : {payload['integrity_status']}")
    print(f"[OK] verified_file_count       : {payload['verified_file_count']}")
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
    print(f"[OK] manifest                  : {payload['manifest_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 146 integrity verifier failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
