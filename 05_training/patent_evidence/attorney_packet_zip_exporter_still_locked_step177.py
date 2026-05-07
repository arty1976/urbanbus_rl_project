from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ARTIFACT_VERSION = "step177_attorney_packet_zip_exporter_still_locked_v1"
BUNDLE_STATUS = "ATTORNEY_PACKET_ZIP_EXPORTER_READY_STILL_LOCKED_NONCLAIM"

EXPECTED_STEP176_STATUS = "ATTORNEY_PACKET_DRY_RUN_EXPORTER_READY_NONCLAIM"

GUARD_FLAGS = {
    "dry_run_only": True,
    "zip_exporter_still_locked": True,
    "export_zip_created": False,
    "zip_creation_allowed": False,
    "export_zip_creation_allowed": False,
    "operator_approval_recorded": False,
    "operator_approval_granted": False,
    "attorney_review_required": True,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
    "legal_novelty_opinion_provided": False,
    "filing_ready_without_attorney_review": False,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path_from_manifest(raw_path: str, dry_run_manifest_path: Path, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # Step 176 usually stores output paths relative to the current project root.
    p1 = project_root / p
    if p1.exists():
        return p1
    # Fallback for tests where manifest paths are relative to its parent.
    p2 = dry_run_manifest_path.parent / p
    return p2


def make_sample_step176_manifest(output_root: Path) -> Tuple[Path, Path]:
    """Create a tiny Step 176-like dry-run manifest and candidate files for self-test."""
    sample_root = output_root / "_sample_project"
    patent_dir = sample_root / "05_training" / "patent_evidence"
    patent_dir.mkdir(parents=True, exist_ok=True)

    sample_files = [
        "05_training/patent_evidence/final_zero_loss_patent_evidence_handoff_index_step172.md",
        "05_training/patent_evidence/zero_loss_patent_claim_drafting_packet_step173.md",
        "05_training/patent_evidence/patent_attorney_review_packet_index_step174.md",
        "05_training/patent_evidence/patent_packet_export_checklist_step175.md",
        "05_training/patent_evidence/patent_evidence_report_v2_schema_step166.md",
        "05_training/patent_evidence/attempt_route_path_attention_filter_schema_step165.md",
    ]
    include_rows: List[Dict[str, Any]] = []
    for idx, rel in enumerate(sample_files, start=1):
        path = sample_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Sample file {idx}\n\n{rel}\n", encoding="utf-8")
        include_rows.append({
            "relative_path": rel,
            "category": "sample_attorney_packet",
            "required": "True",
            "exists": "True",
            "include_in_dry_run": "True",
            "description": "Sample Step 176 dry-run candidate.",
            "size_bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
            "reason": "included_candidate_exists",
        })

    step176_dir = output_root / "_sample_step176"
    step176_dir.mkdir(parents=True, exist_ok=True)
    include_csv = step176_dir / "attorney_packet_dry_run_include_index_step176.csv"
    missing_csv = step176_dir / "attorney_packet_dry_run_missing_index_step176.csv"
    exclusion_csv = step176_dir / "attorney_packet_dry_run_exclusion_index_step176.csv"
    manifest_path = step176_dir / "attorney_packet_dry_run_manifest_step176.json"

    write_csv(
        include_csv,
        include_rows,
        ["relative_path", "category", "required", "exists", "include_in_dry_run", "description", "size_bytes", "sha256", "reason"],
    )
    write_csv(missing_csv, [], ["relative_path", "category", "required", "exists", "include_in_dry_run", "description", "reason"])
    write_csv(
        exclusion_csv,
        [
            {"pattern": "artifacts/patent_evidence/*_selftest/**", "exclude_by_default": True, "reason": "generated self-test artifacts"},
            {"pattern": "05_training/.venv/**", "exclude_by_default": True, "reason": "environment folder"},
            {"pattern": "*.pt", "exclude_by_default": True, "reason": "model binary excluded from attorney packet dry-run"},
        ],
        ["pattern", "exclude_by_default", "reason"],
    )
    manifest = {
        "artifact_version": "step176_attorney_packet_dry_run_exporter_v1",
        "created_at_utc": now_utc(),
        "project_root": str(sample_root),
        "output_root": str(step176_dir),
        "audit_status": "PASS",
        "bundle_status": EXPECTED_STEP176_STATUS,
        "include_candidate_count": len(include_rows),
        "included_existing_count": len(include_rows),
        "missing_candidate_count": 0,
        "missing_required_count": 0,
        "hard_failures": 0,
        "warnings": 0,
        "dry_run_only": True,
        "export_zip_created": False,
        "export_zip_creation_allowed": False,
        "attorney_review_required": True,
        "filing_ready_without_attorney_review": False,
        "output_files": {
            "attorney_packet_dry_run_include_index_csv": str(include_csv),
            "attorney_packet_dry_run_missing_index_csv": str(missing_csv),
            "attorney_packet_dry_run_exclusion_index_csv": str(exclusion_csv),
            "attorney_packet_dry_run_manifest_json": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest_path, sample_root


def validate_step176_manifest(dry_run_manifest_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if not dry_run_manifest_path.exists():
        raise RuntimeError(f"Step 176 dry-run manifest not found: {dry_run_manifest_path}")
    m = load_json(dry_run_manifest_path)
    if m.get("bundle_status") != EXPECTED_STEP176_STATUS:
        raise RuntimeError(f"Step 176 bundle_status mismatch: {m.get('bundle_status')}")
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"Step 176 audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("dry_run_only") is not True:
        raise RuntimeError("Step 176 dry_run_only must be true")
    if m.get("export_zip_created") is not False:
        raise RuntimeError("Step 176 export_zip_created must be false")
    if m.get("export_zip_creation_allowed") is not False:
        raise RuntimeError("Step 176 export_zip_creation_allowed must be false")
    if int(m.get("missing_required_count", -1)) != 0:
        raise RuntimeError(f"Step 176 missing_required_count must be 0, got {m.get('missing_required_count')}")
    if int(m.get("included_existing_count", 0)) <= 0:
        raise RuntimeError("Step 176 included_existing_count must be positive")
    if m.get("attorney_review_required") is not True:
        warnings.append("Step 176 attorney_review_required was not true; Step 177 keeps attorney review required.")
    return m, warnings


def build_zip_plan(step176_manifest: Dict[str, Any], dry_run_manifest_path: Path, project_root: Path, output_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    output_files = step176_manifest.get("output_files", {})
    include_path = resolve_path_from_manifest(
        str(output_files.get("attorney_packet_dry_run_include_index_csv", "")),
        dry_run_manifest_path,
        project_root,
    )
    exclusion_path = resolve_path_from_manifest(
        str(output_files.get("attorney_packet_dry_run_exclusion_index_csv", "")),
        dry_run_manifest_path,
        project_root,
    )

    if not include_path.exists():
        raise RuntimeError(f"Step 176 include index not found: {include_path}")
    include_rows = read_csv_dicts(include_path)
    exclusion_rows = read_csv_dicts(exclusion_path) if exclusion_path.exists() else []

    zip_plan: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for row in include_rows:
        rel = str(row.get("relative_path", "")).strip()
        include = truthy(row.get("include_in_dry_run")) and truthy(row.get("exists"))
        if not include:
            continue
        src_path = project_root / rel
        exists_now = src_path.exists() and src_path.is_file()
        if not exists_now:
            missing = dict(row)
            missing["missing_at_step177"] = True
            missing["reason_step177"] = "candidate existed in Step 176 but is missing at Step 177"
            missing_rows.append(missing)
            continue
        current_sha = sha256_file(src_path)
        step176_sha = str(row.get("sha256", "")).strip()
        sha_matches = bool(step176_sha) and current_sha == step176_sha
        if not sha_matches:
            warnings.append(f"sha256 changed since Step 176: {rel}")
        zip_plan.append({
            "relative_path": rel,
            "zip_internal_path": rel,
            "category": row.get("category", ""),
            "required": row.get("required", ""),
            "size_bytes": src_path.stat().st_size,
            "sha256_step176": step176_sha,
            "sha256_step177": current_sha,
            "sha256_matches_step176": sha_matches,
            "would_include_in_zip_if_released": True,
            "source_path": str(src_path),
        })

    return zip_plan, missing_rows, exclusion_rows, warnings


def generate_report(path: Path, manifest: Dict[str, Any], zip_plan: List[Dict[str, Any]], missing_rows: List[Dict[str, Any]], exclusion_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Step 177 Attorney Packet ZIP Exporter — Still Locked")
    lines.append("")
    lines.append("This report is a ZIP export preflight for the Zero-Loss Pickup patent attorney packet. It does not create a ZIP archive.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    for key in [
        "audit_status",
        "bundle_status",
        "dry_run_only",
        "zip_exporter_still_locked",
        "zip_creation_allowed",
        "export_zip_created",
        "planned_zip_filename",
        "would_include_count",
        "missing_at_step177_count",
        "hard_failures",
        "warnings",
    ]:
        lines.append(f"- `{key}`: `{manifest.get(key)}`")
    lines.append("")
    lines.append("## Would include if later released")
    lines.append("")
    lines.append("| relative_path | category | size_bytes | sha256_matches_step176 |")
    lines.append("|---|---|---:|---:|")
    for row in zip_plan:
        lines.append(f"| `{row['relative_path']}` | {row.get('category','')} | {row.get('size_bytes','')} | {row.get('sha256_matches_step176')} |")
    lines.append("")
    lines.append("## Missing at Step 177")
    lines.append("")
    if missing_rows:
        lines.append("| relative_path | reason |")
        lines.append("|---|---|")
        for row in missing_rows:
            lines.append(f"| `{row.get('relative_path','')}` | {row.get('reason_step177','')} |")
    else:
        lines.append("No Step 177 missing files detected.")
    lines.append("")
    lines.append("## Exclusion patterns inherited from Step 176")
    lines.append("")
    for row in exclusion_rows:
        lines.append(f"- `{row.get('pattern','')}` — {row.get('reason','')}")
    lines.append("")
    lines.append("## Guard statement")
    lines.append("")
    lines.append("ZIP creation remains locked. Attorney review is required. This is not a filing-ready legal package.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exporter(
    project_root: Path,
    output_root: Path,
    dry_run_manifest: Path | None = None,
    mode: str = "from-step176",
    create_zip: bool = False,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    if mode == "sample":
        dry_run_manifest, project_root = make_sample_step176_manifest(output_root)
    elif dry_run_manifest is None:
        raise RuntimeError("--dry-run-manifest is required unless --mode sample is used")

    assert dry_run_manifest is not None
    step176_manifest, warnings1 = validate_step176_manifest(dry_run_manifest)
    # Prefer Step 176's own project_root because included candidate paths were scanned relative to it.
    step176_project_root = Path(str(step176_manifest.get("project_root", project_root)))
    if step176_project_root.exists():
        project_root = step176_project_root

    zip_plan, missing_rows, exclusion_rows, warnings2 = build_zip_plan(step176_manifest, dry_run_manifest, project_root, output_root)
    warnings = warnings1 + warnings2
    missing_at_step177_count = len(missing_rows)

    hard_failures = missing_at_step177_count
    if create_zip:
        # This exporter is explicitly still locked. A create request must not create a ZIP.
        hard_failures += 1
        warnings.append("--create-zip was requested, but Step 177 is still locked and refuses ZIP creation.")

    planned_zip_filename = "zero_loss_patent_attorney_review_packet_STEP177_LOCKED_DRY_RUN.zip"
    planned_zip_path = output_root / planned_zip_filename
    if planned_zip_path.exists():
        hard_failures += 1
        warnings.append(f"planned ZIP path already exists and must be removed before validation: {planned_zip_path}")

    manifest_path = output_root / "attorney_packet_zip_exporter_still_locked_manifest_step177.json"
    output_files = {
        "zip_exporter_report_md": str(output_root / "attorney_packet_zip_exporter_still_locked_report_step177.md"),
        "zip_export_plan_csv": str(output_root / "attorney_packet_zip_export_plan_step177.csv"),
        "zip_export_missing_at_step177_csv": str(output_root / "attorney_packet_zip_export_missing_at_step177.csv"),
        "zip_export_exclusion_plan_csv": str(output_root / "attorney_packet_zip_export_exclusion_plan_step177.csv"),
        "zip_exporter_manifest_json": str(manifest_path),
    }

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "step176_dry_run_manifest": str(dry_run_manifest),
        "audit_status": "PASS" if hard_failures == 0 else "BLOCKED",
        "bundle_status": BUNDLE_STATUS,
        "would_include_count": len(zip_plan),
        "missing_at_step177_count": missing_at_step177_count,
        "exclusion_pattern_count": len(exclusion_rows),
        "planned_zip_filename": planned_zip_filename,
        "planned_zip_path": str(planned_zip_path),
        "hard_failures": int(hard_failures),
        "warnings": int(len(warnings)),
        "warning_messages": warnings,
        "output_files": output_files,
        **GUARD_FLAGS,
    }

    write_csv(
        output_root / "attorney_packet_zip_export_plan_step177.csv",
        zip_plan,
        [
            "relative_path",
            "zip_internal_path",
            "category",
            "required",
            "size_bytes",
            "sha256_step176",
            "sha256_step177",
            "sha256_matches_step176",
            "would_include_in_zip_if_released",
            "source_path",
        ],
    )
    write_csv(
        output_root / "attorney_packet_zip_export_missing_at_step177.csv",
        missing_rows,
        ["relative_path", "category", "required", "exists", "include_in_dry_run", "description", "reason_step177"],
    )
    write_csv(
        output_root / "attorney_packet_zip_export_exclusion_plan_step177.csv",
        exclusion_rows,
        ["pattern", "exclude_by_default", "reason"],
    )
    generate_report(output_root / "attorney_packet_zip_exporter_still_locked_report_step177.md", manifest, zip_plan, missing_rows, exclusion_rows)
    write_json(manifest_path, manifest)

    print("[OK] Step 177 attorney packet ZIP exporter still locked completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] dry_run_only              : {manifest['dry_run_only']}")
    print(f"[OK] zip_exporter_still_locked : {manifest['zip_exporter_still_locked']}")
    print(f"[OK] zip_creation_allowed      : {manifest['zip_creation_allowed']}")
    print(f"[OK] export_zip_created        : {manifest['export_zip_created']}")
    print(f"[OK] would_include_count       : {manifest['would_include_count']}")
    print(f"[OK] missing_at_step177_count  : {manifest['missing_at_step177_count']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] attorney_review_required  : {manifest['attorney_review_required']}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 177 attorney packet ZIP exporter still locked")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run-manifest", default="")
    parser.add_argument("--mode", choices=["from-step176", "sample"], default="from-step176")
    parser.add_argument("--create-zip", action="store_true", help="Intentionally blocked in Step 177")
    args = parser.parse_args()

    run_exporter(
        project_root=Path(args.project_root).resolve(),
        output_root=Path(args.output_root),
        dry_run_manifest=Path(args.dry_run_manifest) if args.dry_run_manifest else None,
        mode=args.mode,
        create_zip=bool(args.create_zip),
    )


if __name__ == "__main__":
    main()
