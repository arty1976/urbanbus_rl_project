from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


BUNDLE_STATUS = "ATTORNEY_PACKET_DRY_RUN_EXPORTER_READY_NONCLAIM"
ARTIFACT_VERSION = "step176_attorney_packet_dry_run_exporter_v1"

GUARD_FLAGS = {
    "dry_run_only": True,
    "export_zip_created": False,
    "export_zip_creation_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
    "legal_novelty_opinion_provided": False,
    "filing_ready_without_attorney_review": False,
    "attorney_review_required": True,
}

DEFAULT_INCLUDE_CANDIDATES = [
    # Core handoff and attorney review layer
    ("05_training/patent_evidence/final_zero_loss_patent_evidence_handoff_index_step172.md", True, "handoff_index", "Final index of Step 160-171 zero-loss evidence pipeline."),
    ("05_training/patent_evidence/zero_loss_patent_claim_drafting_packet_step173.md", True, "claim_drafting", "Technical claim drafting packet for attorney review."),
    ("05_training/patent_evidence/patent_attorney_review_packet_index_step174.md", True, "attorney_review_index", "Attorney-facing review packet index and questions."),
    ("05_training/patent_evidence/patent_packet_export_checklist_step175.md", True, "export_checklist", "Inclusion/exclusion checklist for attorney packet export."),
    # Evidence/reporting layer
    ("05_training/patent_evidence/patent_evidence_report_v2_schema_step166.md", True, "evidence_report_v2", "Schema for patent evidence report v2."),
    ("05_training/patent_evidence/patent_evidence_report_v2_step166.py", True, "evidence_report_v2", "Report v2 generator source."),
    ("05_training/patent_evidence/attempt_route_path_attention_filter_schema_step165.md", True, "route_path_filter", "Attempt-specific route/path attention filtering schema."),
    ("05_training/patent_evidence/attempt_route_path_attention_filter_step165.py", True, "route_path_filter", "Attempt-specific attention filter source."),
    ("05_training/patent_evidence/gatv2_real_attention_extractor_schema_step163.md", True, "real_attention", "Real GATv2 attention extraction schema."),
    ("05_training/patent_evidence/gatv2_real_attention_extractor_step163.py", True, "real_attention", "Real GATv2 attention extractor source."),
    ("05_training/patent_evidence/zero_loss_pickup_evidence_schema_step160.md", True, "zero_loss_reporter", "Zero-loss pickup evidence schema."),
    ("05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py", True, "zero_loss_reporter", "Zero-loss pickup reporter source."),
    # Runbook / execution gate layer
    ("05_training/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167.md", False, "actual_runbook", "Actual-like rollout evidence runbook."),
    ("05_training/patent_evidence/actual_evidence_input_readiness_checklist_step169.md", False, "readiness", "Input readiness checklist."),
    ("05_training/patent_evidence/actual_evidence_command_packet_step170.md", False, "command_packet", "Dry-run command packet."),
    ("05_training/patent_evidence/actual_evidence_execution_release_checklist_step171.md", False, "release_gate", "Execution release checklist."),
    # Project context references
    ("project_log.md", False, "project_context", "Project log containing Step 160-168 update."),
    ("H200_research_runbook_final.md", False, "project_context", "H200 runbook if present at project root."),
]

EXCLUSION_PATTERNS = [
    "artifacts/patent_evidence/*_selftest/**",
    "artifacts/patent_evidence/*integration_selftest/**",
    "artifacts/rewards/**",
    "*.latest.json",
    "05_training/.venv/**",
    "__pycache__/**",
    ".pytest_cache/**",
    ".git/**",
    "*.pt",
    "*.parquet",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def stable_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def scan_candidates(project_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    include_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []

    for rel, required, category, description in DEFAULT_INCLUDE_CANDIDATES:
        path = project_root / rel
        exists = path.exists()
        row: Dict[str, Any] = {
            "relative_path": rel,
            "category": category,
            "required": bool(required),
            "exists": bool(exists),
            "include_in_dry_run": bool(exists),
            "description": description,
            "size_bytes": int(path.stat().st_size) if exists and path.is_file() else "",
            "sha256": sha256_file(path) if exists and path.is_file() else "",
            "reason": "included_candidate_exists" if exists else ("missing_required" if required else "missing_optional"),
        }
        include_rows.append(row)
        if not exists:
            missing_rows.append(row)

    for pattern in EXCLUSION_PATTERNS:
        excluded_rows.append({
            "pattern": pattern,
            "exclude_by_default": True,
            "reason": "avoid generated self-test artifacts, volatile outputs, large binary data, or environment folders",
        })

    return include_rows, missing_rows, excluded_rows


def generate_report_md(path: Path, manifest: Dict[str, Any], include_rows: List[Dict[str, Any]], missing_rows: List[Dict[str, Any]], excluded_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Step 176 Attorney Packet Dry-Run Exporter")
    lines.append("")
    lines.append("This document is a dry-run export manifest for the Zero-Loss Pickup patent attorney review packet.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    for key in [
        "audit_status",
        "bundle_status",
        "dry_run_only",
        "export_zip_created",
        "export_zip_creation_allowed",
        "attorney_review_required",
        "filing_ready_without_attorney_review",
        "hard_failures",
        "warnings",
    ]:
        lines.append(f"- `{key}`: `{manifest.get(key)}`")
    lines.append("")
    lines.append("## Included candidates")
    lines.append("")
    lines.append("| relative_path | category | required | exists | include |")
    lines.append("|---|---|---:|---:|---:|")
    for row in include_rows:
        lines.append(f"| `{row['relative_path']}` | {row['category']} | {row['required']} | {row['exists']} | {row['include_in_dry_run']} |")
    lines.append("")
    lines.append("## Missing candidates")
    lines.append("")
    if missing_rows:
        lines.append("| relative_path | required | reason |")
        lines.append("|---|---:|---|")
        for row in missing_rows:
            lines.append(f"| `{row['relative_path']}` | {row['required']} | {row['reason']} |")
    else:
        lines.append("No missing candidate files detected.")
    lines.append("")
    lines.append("## Exclusion patterns")
    lines.append("")
    for row in excluded_rows:
        lines.append(f"- `{row['pattern']}` — {row['reason']}")
    lines.append("")
    lines.append("## Guard statement")
    lines.append("")
    lines.append("This step does not create a ZIP archive, does not authorize filing, and does not make legal novelty or performance claims.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exporter(project_root: Path, output_root: Path, allow_missing_required: bool = False) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    include_rows, missing_rows, excluded_rows = scan_candidates(project_root)

    missing_required = [r for r in missing_rows if bool(r.get("required"))]
    hard_failures = 0 if allow_missing_required else len(missing_required)
    warnings = len([r for r in missing_rows if not bool(r.get("required"))])

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "audit_status": "PASS" if hard_failures == 0 else "BLOCKED",
        "bundle_status": BUNDLE_STATUS,
        "include_candidate_count": len(include_rows),
        "included_existing_count": sum(1 for r in include_rows if bool(r["exists"])),
        "missing_candidate_count": len(missing_rows),
        "missing_required_count": len(missing_required),
        "exclusion_pattern_count": len(excluded_rows),
        "hard_failures": int(hard_failures),
        "warnings": int(warnings),
        "allow_missing_required": bool(allow_missing_required),
        "output_files": {
            "attorney_packet_dry_run_report_md": str(output_root / "attorney_packet_dry_run_report_step176.md"),
            "attorney_packet_dry_run_include_index_csv": str(output_root / "attorney_packet_dry_run_include_index_step176.csv"),
            "attorney_packet_dry_run_missing_index_csv": str(output_root / "attorney_packet_dry_run_missing_index_step176.csv"),
            "attorney_packet_dry_run_exclusion_index_csv": str(output_root / "attorney_packet_dry_run_exclusion_index_step176.csv"),
            "attorney_packet_dry_run_manifest_json": str(output_root / "attorney_packet_dry_run_manifest_step176.json"),
        },
        **GUARD_FLAGS,
    }

    write_csv(
        output_root / "attorney_packet_dry_run_include_index_step176.csv",
        include_rows,
        ["relative_path", "category", "required", "exists", "include_in_dry_run", "description", "size_bytes", "sha256", "reason"],
    )
    write_csv(
        output_root / "attorney_packet_dry_run_missing_index_step176.csv",
        missing_rows,
        ["relative_path", "category", "required", "exists", "include_in_dry_run", "description", "reason"],
    )
    write_csv(
        output_root / "attorney_packet_dry_run_exclusion_index_step176.csv",
        excluded_rows,
        ["pattern", "exclude_by_default", "reason"],
    )
    generate_report_md(output_root / "attorney_packet_dry_run_report_step176.md", manifest, include_rows, missing_rows, excluded_rows)
    write_json(output_root / "attorney_packet_dry_run_manifest_step176.json", manifest)

    print("[OK] Step 176 attorney packet dry-run exporter completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] dry_run_only              : {manifest['dry_run_only']}")
    print(f"[OK] include_candidate_count   : {manifest['include_candidate_count']}")
    print(f"[OK] included_existing_count   : {manifest['included_existing_count']}")
    print(f"[OK] missing_required_count    : {manifest['missing_required_count']}")
    print(f"[OK] export_zip_created        : {manifest['export_zip_created']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {output_root / 'attorney_packet_dry_run_manifest_step176.json'}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 176 attorney packet dry-run exporter")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/attorney_packet_dry_run_exporter_step176_selftest")
    parser.add_argument("--allow-missing-required", action="store_true")
    args = parser.parse_args()

    run_exporter(
        project_root=Path(args.project_root).resolve(),
        output_root=Path(args.output_root),
        allow_missing_required=bool(args.allow_missing_required),
    )


if __name__ == "__main__":
    main()
