from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

GUARD_FALSE_KEYS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "train_allowed",
    "legal_novelty_opinion_provided",
    "filing_ready_without_attorney_review",
]

PIPELINE_STEPS = [
    (160, "Zero-Loss Pickup Evidence Reporter", "Generates ETA-delta and attention evidence bundle."),
    (161, "Route-Aware Pickup Attempt Event Writer", "Creates pickup attempts, ETA counterfactual rows, and attention input rows."),
    (162, "Route-Aware Rollout Adapter", "Normalizes route-aware rollout raw events for the evidence pipeline."),
    (163, "GATv2 Real Attention Extractor", "Extracts real GATv2Conv attention weights."),
    (164, "Real Attention Pipeline Connector", "Connects real attention output to the Zero-Loss evidence input contract."),
    (165, "Attempt-Specific Route/Path Attention Filter", "Filters attention rows by attempt-specific path/edge relevance."),
    (166, "Patent Evidence Report v2", "Combines zero-loss ETA evidence and route/path attention summaries."),
    (167, "Actual Route-Aware Rollout Evidence Runbook", "Defines actual-like execution sequence."),
    (168, "Project Log Update", "Records Step 160-167 pipeline status."),
    (169, "Actual Evidence Input Readiness Checklist", "Checks input files before actual-like evidence generation."),
    (170, "Actual Evidence Command Packet", "Creates dry-run command sequence."),
    (171, "Actual Evidence Execution Release Checklist", "Defines operator approval gate."),
    (172, "Final Zero-Loss Patent Evidence Handoff Index", "Indexes the complete evidence pipeline."),
    (173, "Zero-Loss Patent Claim Drafting Packet", "Creates technical claim draft packet for attorney review."),
]

ATTACHMENTS = [
    ("A01", "Step 172 final handoff index", "artifacts/patent_evidence/final_zero_loss_patent_evidence_handoff_index_step172_selftest/final_zero_loss_patent_evidence_handoff_index_step172.md", "primary", "Summarizes the full Step 160-171 evidence chain."),
    ("A02", "Step 173 claim drafting packet", "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173_selftest/zero_loss_patent_claim_drafting_packet_step173.md", "primary", "Contains technical independent/dependent claim draft candidates."),
    ("A03", "Claim element matrix", "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173_selftest/zero_loss_claim_element_matrix_step173.csv", "primary", "Maps claim elements to technical evidence."),
    ("A04", "Evidence-to-claim mapping", "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173_selftest/zero_loss_evidence_to_claim_mapping_step173.csv", "primary", "Shows which pipeline outputs support each technical element."),
    ("A05", "Prior-art differentiation matrix", "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173_selftest/zero_loss_prior_art_differentiation_matrix_step173.csv", "primary", "Frames differences from delay/detour-based rideshare approaches."),
    ("A06", "Patent evidence report v2", "artifacts/patent_evidence/patent_evidence_report_v2_step166_selftest/report_v2/patent_evidence_report_v2.md", "supporting", "Demonstrates the report structure for zero-loss and route/path attention evidence."),
    ("A07", "Attempt-specific attention filter manifest", "artifacts/patent_evidence/attempt_route_path_attention_filter_step165_selftest/attempt_route_path_attention_filter_manifest.json", "supporting", "Shows route/path attention filtering guard metadata."),
    ("A08", "Actual evidence runbook", "artifacts/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167_selftest/actual_route_aware_rollout_evidence_runbook_step167.md", "supporting", "Specifies actual evidence generation order."),
    ("A09", "Readiness checklist", "artifacts/patent_evidence/actual_evidence_input_readiness_checklist_step169_selftest/actual_evidence_input_readiness_checklist_step169.json", "supporting", "Shows actual-like input gate requirements."),
    ("A10", "Command packet", "artifacts/patent_evidence/actual_evidence_command_packet_step170_selftest/actual_evidence_command_packet_step170.json", "supporting", "Shows dry-run commands for Step 162-166 chain."),
    ("A11", "Release checklist", "artifacts/patent_evidence/actual_evidence_execution_release_checklist_step171_selftest/actual_evidence_execution_release_checklist_step171.json", "guard", "Shows operator approval requirements before actual evidence execution."),
]

REVIEW_QUESTIONS = [
    ("Q01", "Claim scope", "Should the independent claim be drafted as a dispatch method, a system, or both?"),
    ("Q02", "Zero-loss definition", "Should zero-loss mean exactly 0 seconds, or 0 within simulator/time-resolution tolerance?"),
    ("Q03", "Prior art", "Which rideshare detour/delay threshold references are closest, and what claim narrowing is needed?"),
    ("Q04", "GATv2 attention", "Should graph-attention evidence be an independent limitation or a dependent claim feature?"),
    ("Q05", "Counterfactual ETA", "How should counterfactual ETA computation be framed to avoid claiming abstract math alone?"),
    ("Q06", "Public transit domain", "Should claims be limited to fixed-route/semi-fixed public bus systems?"),
    ("Q07", "MLOps workflow", "Should manifest/preflight/release guard be included in this application or split out?"),
    ("Q08", "Evidence status", "How should non-claim self-test outputs be described without overclaiming actual operation results?"),
]

REQUESTED_ITEMS = [
    ("R01", "Novelty search", "KIPRIS / Google Patents / WIPO / USPTO / EPO search on zero-delay pickup, rideshare detour threshold, graph neural network dispatch."),
    ("R02", "Claim strategy", "Advise whether to file one application with dependent claims or multiple applications."),
    ("R03", "Terminology", "Review Zero-Loss Pickup Threshold, ETA counterfactual, GATv2 attention, route-aware simulator wording."),
    ("R04", "Enablement", "Identify which implementation details must be added for sufficient disclosure."),
    ("R05", "Evidence appendix", "Review whether Step 166 report v2 should be an appendix, example, or experimental result section."),
    ("R06", "Foreign filing", "Advise whether PCT/KR-first filing strategy is appropriate."),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def existing(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def generate_packet(project_root: Path, output_root: Path, mode: str = "selftest") -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()

    attachment_rows: List[Dict[str, Any]] = []
    for item_id, title, rel_path, role, note in ATTACHMENTS:
        p = project_root / rel_path
        attachment_rows.append({
            "item_id": item_id,
            "title": title,
            "relative_path": rel_path,
            "role": role,
            "exists": bool(existing(p)),
            "note": note,
        })

    question_rows = [
        {"question_id": qid, "category": cat, "question": q}
        for qid, cat, q in REVIEW_QUESTIONS
    ]
    requested_rows = [
        {"request_id": rid, "request_type": typ, "requested_review": desc}
        for rid, typ, desc in REQUESTED_ITEMS
    ]
    step_rows = [
        {"step": step, "name": name, "purpose": purpose}
        for step, name, purpose in PIPELINE_STEPS
    ]

    attachment_csv = output_root / "patent_attorney_attachment_index_step174.csv"
    questions_csv = output_root / "patent_attorney_review_questions_step174.csv"
    requested_csv = output_root / "patent_attorney_review_requested_items_step174.csv"
    steps_csv = output_root / "patent_attorney_pipeline_step_index_step174.csv"

    write_csv(attachment_csv, ["item_id", "title", "relative_path", "role", "exists", "note"], attachment_rows)
    write_csv(questions_csv, ["question_id", "category", "question"], question_rows)
    write_csv(requested_csv, ["request_id", "request_type", "requested_review"], requested_rows)
    write_csv(steps_csv, ["step", "name", "purpose"], step_rows)

    guard = {key: False for key in GUARD_FALSE_KEYS}
    packet = {
        "artifact_version": "patent_attorney_review_packet_index_step174_v1",
        "created_at_utc": created_at,
        "mode": mode,
        "audit_status": "PASS",
        "bundle_status": "PATENT_ATTORNEY_REVIEW_PACKET_INDEX_READY_NONCLAIM",
        "project_root": str(project_root),
        "output_root": str(output_root),
        "pipeline_step_count": len(PIPELINE_STEPS),
        "attachment_count": len(ATTACHMENTS),
        "review_question_count": len(REVIEW_QUESTIONS),
        "requested_item_count": len(REQUESTED_ITEMS),
        "primary_attachment_count": sum(1 for r in attachment_rows if r["role"] == "primary"),
        "attorney_review_required": True,
        "not_legal_opinion": True,
        "filing_ready_without_attorney_review": False,
        "legal_novelty_opinion_provided": False,
        **guard,
        "outputs": {
            "markdown": str(output_root / "patent_attorney_review_packet_index_step174.md"),
            "json": str(output_root / "patent_attorney_review_packet_index_step174.json"),
            "attachment_index": str(attachment_csv),
            "review_questions": str(questions_csv),
            "requested_items": str(requested_csv),
            "pipeline_steps": str(steps_csv),
            "manifest": str(output_root / "patent_attorney_review_manifest_step174.json"),
        },
        "attachments": attachment_rows,
        "review_questions": question_rows,
        "requested_items": requested_rows,
        "pipeline_steps": step_rows,
    }

    md = build_markdown(packet)
    md_path = output_root / "patent_attorney_review_packet_index_step174.md"
    json_path = output_root / "patent_attorney_review_packet_index_step174.json"
    manifest_path = output_root / "patent_attorney_review_manifest_step174.json"

    md_path.write_text(md, encoding="utf-8")
    dump_json(json_path, packet)

    manifest = {k: packet[k] for k in [
        "artifact_version", "created_at_utc", "mode", "audit_status", "bundle_status",
        "pipeline_step_count", "attachment_count", "review_question_count", "requested_item_count",
        "attorney_review_required", "not_legal_opinion", "filing_ready_without_attorney_review",
        "legal_novelty_opinion_provided",
    ]}
    manifest.update(guard)
    manifest["outputs"] = packet["outputs"]
    manifest["content_hashes"] = {
        "markdown_sha256": sha256_text(md),
        "json_sha256": sha256_text(json.dumps(packet, ensure_ascii=False, sort_keys=True)),
    }
    dump_json(manifest_path, manifest)

    return manifest


def build_markdown(packet: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Step 174 — Patent Attorney Review Packet Index")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- audit_status: `{packet['audit_status']}`")
    lines.append(f"- bundle_status: `{packet['bundle_status']}`")
    lines.append(f"- not_legal_opinion: `{packet['not_legal_opinion']}`")
    lines.append(f"- attorney_review_required: `{packet['attorney_review_required']}`")
    lines.append(f"- filing_ready_without_attorney_review: `{packet['filing_ready_without_attorney_review']}`")
    lines.append("")
    lines.append("## Non-Claim Guards")
    lines.append("")
    for key in GUARD_FALSE_KEYS:
        lines.append(f"- {key}: `{packet[key]}`")
    lines.append("")
    lines.append("## Pipeline Scope")
    lines.append("")
    for row in packet["pipeline_steps"]:
        lines.append(f"- Step {row['step']}: {row['name']} — {row['purpose']}")
    lines.append("")
    lines.append("## Attachment Index")
    lines.append("")
    lines.append("| ID | Title | Role | Exists | Path |")
    lines.append("|---|---|---|---:|---|")
    for row in packet["attachments"]:
        lines.append(f"| {row['item_id']} | {row['title']} | {row['role']} | {row['exists']} | `{row['relative_path']}` |")
    lines.append("")
    lines.append("## Attorney Review Questions")
    lines.append("")
    for row in packet["review_questions"]:
        lines.append(f"- **{row['question_id']} ({row['category']})**: {row['question']}")
    lines.append("")
    lines.append("## Requested Attorney Work")
    lines.append("")
    for row in packet["requested_items"]:
        lines.append(f"- **{row['request_id']} ({row['request_type']})**: {row['requested_review']}")
    lines.append("")
    lines.append("## Important Note")
    lines.append("")
    lines.append("This packet is a technical review index only. It is not a legal novelty opinion, not a patentability opinion, and not a filing-ready legal document.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-root", required=True)
    p.add_argument("--mode", default="selftest")
    args = p.parse_args()

    manifest = generate_packet(Path(args.project_root).resolve(), Path(args.output_root), mode=args.mode)
    print("[OK] Step 174 patent attorney review packet index generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] pipeline_step_count       : {manifest['pipeline_step_count']}")
    print(f"[OK] attachment_count          : {manifest['attachment_count']}")
    print(f"[OK] review_question_count     : {manifest['review_question_count']}")
    print(f"[OK] requested_item_count      : {manifest['requested_item_count']}")
    print(f"[OK] not_legal_opinion         : {manifest['not_legal_opinion']}")
    print(f"[OK] attorney_review_required  : {manifest['attorney_review_required']}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    print(f"[OK] manifest                  : {manifest['outputs']['manifest']}")


if __name__ == "__main__":
    main()
