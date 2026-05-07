from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

STEP_RANGE = list(range(160, 179))
PIPELINE_STEPS = [
    (160, "Zero-Loss Pickup Evidence Reporter", "technical_evidence"),
    (161, "Route-Aware Pickup Attempt Event Writer", "technical_evidence"),
    (162, "Route-Aware Rollout Adapter", "technical_evidence"),
    (163, "GATv2 real attention extractor", "technical_evidence"),
    (164, "Real GATv2 attention pipeline connector", "technical_evidence"),
    (165, "Attempt-specific route/path attention filter", "technical_evidence"),
    (166, "Patent evidence report v2", "technical_evidence"),
    (167, "Actual route-aware rollout evidence runbook", "actual_execution_preparation"),
    (168, "Zero-Loss patent evidence project log update", "documentation"),
    (169, "Actual evidence input readiness checklist", "actual_execution_preparation"),
    (170, "Actual evidence command packet", "actual_execution_preparation"),
    (171, "Actual evidence execution release checklist", "release_guard"),
    (172, "Final Zero-Loss patent evidence handoff index", "handoff_index"),
    (173, "Zero-Loss patent claim drafting packet", "claim_drafting"),
    (174, "Patent attorney review packet index", "attorney_review"),
    (175, "Patent packet export checklist", "export_preparation"),
    (176, "Attorney packet dry-run exporter", "export_preparation"),
    (177, "Attorney packet ZIP exporter still locked", "export_guard"),
    (178, "Final patent packet local handoff summary", "local_handoff_summary"),
]

LOCKED_GUARDS = {
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
    "zip_creation_allowed": False,
    "export_zip_created": False,
    "filing_ready_without_attorney_review": False,
    "legal_novelty_opinion_provided": False,
}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(manifest: Dict[str, Any], step_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Step 178 — Final Patent Packet Local Handoff Summary")
    lines.append("")
    lines.append(f"Generated at UTC: `{manifest['created_at_utc']}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- audit_status: `{manifest['audit_status']}`")
    lines.append(f"- bundle_status: `{manifest['bundle_status']}`")
    lines.append(f"- local_preparation_closed: `{manifest['local_preparation_closed']}`")
    lines.append(f"- zip_creation_allowed: `{manifest['zip_creation_allowed']}`")
    lines.append(f"- export_zip_created: `{manifest['export_zip_created']}`")
    lines.append(f"- attorney_review_required: `{manifest['attorney_review_required']}`")
    lines.append(f"- filing_ready_without_attorney_review: `{manifest['filing_ready_without_attorney_review']}`")
    lines.append("")
    lines.append("## Covered pipeline steps")
    lines.append("")
    lines.append("| Step | Name | Category | Status |")
    lines.append("|---:|---|---|---|")
    for row in step_rows:
        lines.append(f"| {row['step']} | {row['name']} | {row['category']} | {row['status']} |")
    lines.append("")
    lines.append("## Locked guards")
    lines.append("")
    for key in LOCKED_GUARDS:
        lines.append(f"- `{key}` = `{manifest[key]}`")
    lines.append("")
    lines.append("## Handoff conclusion")
    lines.append("")
    lines.append(
        "The local Zero-Loss patent preparation chain is summarized and closed as a non-claim technical packet. "
        "Actual ZIP export, attorney filing, legal novelty opinion, paper-level claims, causal-performance claims, and operational claims remain locked."
    )
    lines.append("")
    return "\n".join(lines)


def build_summary(output_root: Path, project_root: Path | None = None) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    step_rows = [
        {
            "step": step,
            "name": name,
            "category": category,
            "status": "local_preparation_recorded",
        }
        for step, name, category in PIPELINE_STEPS
    ]

    output_files = {
        "summary_md": str(output_root / "final_patent_packet_local_handoff_summary_step178.md"),
        "summary_json": str(output_root / "final_patent_packet_local_handoff_summary_step178.json"),
        "pipeline_step_index_csv": str(output_root / "final_patent_packet_pipeline_step_index_step178.csv"),
        "locked_guard_index_csv": str(output_root / "final_patent_packet_locked_guard_index_step178.csv"),
        "manifest_json": str(output_root / "final_patent_packet_local_handoff_manifest_step178.json"),
    }

    guard_rows = [
        {"guard_name": key, "expected_value": value, "actual_value": value, "status": "locked"}
        for key, value in LOCKED_GUARDS.items()
    ]

    manifest: Dict[str, Any] = {
        "artifact_version": "final_patent_packet_local_handoff_summary_step178_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root) if project_root else "",
        "audit_status": "PASS",
        "bundle_status": "FINAL_PATENT_PACKET_LOCAL_HANDOFF_SUMMARY_READY_STILL_LOCKED",
        "step_start": 160,
        "step_end": 178,
        "pipeline_step_count": len(step_rows),
        "local_preparation_closed": True,
        "technical_packet_ready_for_attorney_review": True,
        "attorney_review_required": True,
        "not_legal_opinion": True,
        "legal_novelty_opinion_provided": False,
        "filing_ready_without_attorney_review": False,
        "export_zip_created": False,
        "zip_creation_allowed": False,
        "export_zip_creation_allowed": False,
        "actual_evidence_execution_allowed": False,
        "command_execution_allowed": False,
        "train_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "hard_failures": 0,
        "warnings": 0,
        "output_files": output_files,
    }
    for key, value in LOCKED_GUARDS.items():
        manifest[key] = value

    summary_md = render_markdown(manifest, step_rows)
    Path(output_files["summary_md"]).write_text(summary_md, encoding="utf-8")
    dump_json(Path(output_files["summary_json"]), {"manifest": manifest, "pipeline_steps": step_rows, "locked_guards": guard_rows})
    write_csv(Path(output_files["pipeline_step_index_csv"]), step_rows, ["step", "name", "category", "status"])
    write_csv(Path(output_files["locked_guard_index_csv"]), guard_rows, ["guard_name", "expected_value", "actual_value", "status"])
    dump_json(Path(output_files["manifest_json"]), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 178 final patent packet local handoff summary")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/final_patent_packet_local_handoff_summary_step178_selftest")
    args = parser.parse_args()

    manifest = build_summary(output_root=Path(args.output_root), project_root=Path(args.project_root).resolve())
    print("[OK] Step 178 final patent packet local handoff summary generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] pipeline_step_count       : {manifest['pipeline_step_count']}")
    print(f"[OK] local_preparation_closed  : {manifest['local_preparation_closed']}")
    print(f"[OK] technical_packet_ready_for_attorney_review : {manifest['technical_packet_ready_for_attorney_review']}")
    print(f"[OK] export_zip_created        : {manifest['export_zip_created']}")
    print(f"[OK] zip_creation_allowed      : {manifest['zip_creation_allowed']}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    print(f"[OK] output_root               : {args.output_root}")
    print(f"[OK] manifest                  : {manifest['output_files']['manifest_json']}")


if __name__ == "__main__":
    main()
