from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ARTIFACT_VERSION = "patent_packet_export_checklist_step175_v1"
BUNDLE_STATUS = "PATENT_PACKET_EXPORT_CHECKLIST_READY_NONCLAIM"

GUARD_FLAGS: Dict[str, bool] = {
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
    "legal_novelty_opinion_provided": False,
    "filing_ready_without_attorney_review": False,
}


@dataclass(frozen=True)
class IncludeItem:
    item_id: str
    path: str
    category: str
    priority: str
    reason: str
    required_for_attorney_review: bool
    include_in_default_export: bool


@dataclass(frozen=True)
class ExclusionItem:
    pattern: str
    reason: str
    severity: str


@dataclass(frozen=True)
class ReviewQuestion:
    question_id: str
    topic: str
    question: str
    target_reviewer: str
    blocking_before_filing: bool


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_include_items() -> List[IncludeItem]:
    return [
        IncludeItem(
            "I001",
            "artifacts/patent_evidence/final_zero_loss_patent_evidence_handoff_index_step172*/final_zero_loss_patent_evidence_handoff_index_step172.md",
            "handoff_index",
            "required",
            "Final technical index of Steps 160-171 evidence pipeline.",
            True,
            True,
        ),
        IncludeItem(
            "I002",
            "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173*/zero_loss_patent_claim_drafting_packet_step173.md",
            "claim_drafting",
            "required",
            "Technical claim drafting packet for attorney review.",
            True,
            True,
        ),
        IncludeItem(
            "I003",
            "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173*/zero_loss_claim_element_matrix_step173.csv",
            "claim_matrix",
            "required",
            "Maps independent claim elements to system components.",
            True,
            True,
        ),
        IncludeItem(
            "I004",
            "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173*/zero_loss_evidence_to_claim_mapping_step173.csv",
            "evidence_mapping",
            "required",
            "Maps evidence pipeline steps to claim elements.",
            True,
            True,
        ),
        IncludeItem(
            "I005",
            "artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173*/zero_loss_prior_art_differentiation_matrix_step173.csv",
            "prior_art_differentiation",
            "required",
            "Technical differentiation matrix for attorney novelty review.",
            True,
            True,
        ),
        IncludeItem(
            "I006",
            "artifacts/patent_evidence/patent_attorney_review_packet_index_step174*/patent_attorney_review_packet_index_step174.md",
            "attorney_index",
            "required",
            "Attorney-facing review packet index and questions.",
            True,
            True,
        ),
        IncludeItem(
            "I007",
            "artifacts/patent_evidence/patent_attorney_review_packet_index_step174*/patent_attorney_attachment_index_step174.csv",
            "attorney_attachment_index",
            "required",
            "Attachment index to guide review.",
            True,
            True,
        ),
        IncludeItem(
            "I008",
            "05_training/patent_evidence/*_schema_step16*.md",
            "source_schema",
            "recommended",
            "Source schemas for evidence generation and validation.",
            True,
            True,
        ),
        IncludeItem(
            "I009",
            "05_training/patent_evidence/*runbook_step167.md",
            "runbook",
            "recommended",
            "Execution-order runbook for actual evidence generation.",
            True,
            True,
        ),
        IncludeItem(
            "I010",
            "project_log.md",
            "project_log",
            "recommended",
            "Project journal record of the Zero-Loss evidence pipeline.",
            True,
            True,
        ),
        IncludeItem(
            "I011",
            "artifacts/patent_evidence/patent_evidence_report_v2_step166*/report_v2/patent_evidence_report_v2.md",
            "report_v2",
            "optional_when_generated_from_actual_like_inputs",
            "Include only when generated from approved actual-like evidence inputs, not from self-test samples.",
            False,
            False,
        ),
    ]


def default_exclusion_items() -> List[ExclusionItem]:
    return [
        ExclusionItem("artifacts/patent_evidence/*_selftest/**", "Generated self-test artifacts are not attorney evidence by default.", "required"),
        ExclusionItem("artifacts/patent_evidence/*_integration_selftest/**", "Integration smoke outputs are not actual-like evidence by default.", "required"),
        ExclusionItem("**/__pycache__/**", "Python cache folders add noise.", "required"),
        ExclusionItem("**/.pytest_cache/**", "Test cache folders add noise.", "required"),
        ExclusionItem("05_training/.venv/**", "Local Python environment must not be exported.", "required"),
        ExclusionItem("**/*.latest.json", "Machine-local latest pointers are not source of truth.", "required"),
        ExclusionItem("**/*secret*", "Potential secret material must be reviewed and excluded.", "required"),
        ExclusionItem("**/*api_key*", "Potential API key material must be excluded.", "required"),
        ExclusionItem("**/*.pt", "Large model checkpoints require separate hash-only or sealed-evidence handling.", "recommended"),
        ExclusionItem("**/*.pth", "Large model checkpoints require separate hash-only or sealed-evidence handling.", "recommended"),
    ]


def default_review_questions() -> List[ReviewQuestion]:
    return [
        ReviewQuestion("Q001", "claim_scope", "Should the independent claim be framed as a computer-implemented method, system, non-transitory medium, or all three?", "patent_attorney", True),
        ReviewQuestion("Q002", "novelty", "Is the zero-loss ETA threshold plus real GATv2 attention evidence materially distinguishable from delay-threshold ride-sharing prior art?", "patent_attorney", True),
        ReviewQuestion("Q003", "enablement", "Does the evidence pipeline sufficiently support enablement without actual operational deployment results?", "patent_attorney", True),
        ReviewQuestion("Q004", "claim_language", "Should GATv2 be claimed explicitly or generalized to graph attention neural encoder with GATv2 as an embodiment?", "patent_attorney", True),
        ReviewQuestion("Q005", "evidence", "Which self-test artifacts should be excluded from the filing package and which actual-like outputs should be preserved under hash?", "patent_attorney", True),
        ReviewQuestion("Q006", "prior_art", "Which KIPRIS, Google Patents, WIPO, USPTO, and EPO searches should be performed before filing?", "patent_attorney", True),
    ]


def render_markdown(include_items: List[IncludeItem], exclusions: List[ExclusionItem], questions: List[ReviewQuestion], manifest: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Step 175 Patent Packet Export Checklist")
    lines.append("")
    lines.append("This checklist defines the technical export packet for attorney review of the Zero-Loss Pickup Threshold invention.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- audit_status: `{manifest['audit_status']}`")
    lines.append(f"- bundle_status: `{manifest['bundle_status']}`")
    lines.append(f"- include_candidate_count: `{manifest['include_candidate_count']}`")
    lines.append(f"- exclusion_pattern_count: `{manifest['exclusion_pattern_count']}`")
    lines.append(f"- attorney_review_required: `{manifest['attorney_review_required']}`")
    lines.append(f"- export_zip_created: `{manifest['export_zip_created']}`")
    lines.append("")
    lines.append("## Include candidates")
    lines.append("")
    lines.append("| ID | Priority | Category | Include default | Path | Reason |")
    lines.append("|---|---|---|---:|---|---|")
    for item in include_items:
        lines.append(f"| {item.item_id} | {item.priority} | {item.category} | {item.include_in_default_export} | `{item.path}` | {item.reason} |")
    lines.append("")
    lines.append("## Exclusions")
    lines.append("")
    lines.append("| Pattern | Severity | Reason |")
    lines.append("|---|---|---|")
    for item in exclusions:
        lines.append(f"| `{item.pattern}` | {item.severity} | {item.reason} |")
    lines.append("")
    lines.append("## Attorney review questions")
    lines.append("")
    lines.append("| ID | Topic | Blocking | Question |")
    lines.append("|---|---|---:|---|")
    for q in questions:
        lines.append(f"| {q.question_id} | {q.topic} | {q.blocking_before_filing} | {q.question} |")
    lines.append("")
    lines.append("## Non-claim guard")
    lines.append("")
    for key, value in GUARD_FLAGS.items():
        lines.append(f"- {key} = `{value}`")
    lines.append("")
    lines.append("This checklist is not a legal opinion and does not authorize filing without attorney review.")
    lines.append("")
    return "\n".join(lines)


def build_manifest(output_root: Path, mode: str, include_items: List[IncludeItem], exclusions: List[ExclusionItem], questions: List[ReviewQuestion]) -> Dict[str, Any]:
    default_includes = [x for x in include_items if x.include_in_default_export]
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": BUNDLE_STATUS,
        "mode": mode,
        "template_only": True,
        "export_zip_created": False,
        "export_zip_creation_allowed": False,
        "attorney_review_required": True,
        "not_legal_opinion": True,
        "filing_ready_without_attorney_review": False,
        "include_candidate_count": len(include_items),
        "default_include_count": len(default_includes),
        "exclusion_pattern_count": len(exclusions),
        "review_question_count": len(questions),
        "hard_failures": 0,
        "warnings": [
            "self-test and integration artifacts are excluded by default",
            "actual-like evidence outputs require a later release checklist before inclusion",
            "large checkpoints should be represented by hash/reference unless attorney requests sealed copies",
        ],
        "output_root": str(output_root),
        **GUARD_FLAGS,
    }
    return payload


def run(output_root: Path, mode: str = "sample") -> Dict[str, Any]:
    include_items = default_include_items()
    exclusions = default_exclusion_items()
    questions = default_review_questions()
    manifest = build_manifest(output_root, mode, include_items, exclusions, questions)

    output_root.mkdir(parents=True, exist_ok=True)

    include_path = output_root / "patent_packet_export_include_index_step175.csv"
    exclusion_path = output_root / "patent_packet_export_exclusion_index_step175.csv"
    questions_path = output_root / "patent_packet_export_open_questions_step175.csv"
    checklist_md_path = output_root / "patent_packet_export_checklist_step175.md"
    checklist_json_path = output_root / "patent_packet_export_checklist_step175.json"
    manifest_path = output_root / "patent_packet_export_manifest_step175.json"

    write_csv(include_path, [asdict(x) for x in include_items], list(asdict(include_items[0]).keys()))
    write_csv(exclusion_path, [asdict(x) for x in exclusions], list(asdict(exclusions[0]).keys()))
    write_csv(questions_path, [asdict(x) for x in questions], list(asdict(questions[0]).keys()))

    manifest["output_files"] = {
        "include_index": str(include_path),
        "exclusion_index": str(exclusion_path),
        "open_questions": str(questions_path),
        "checklist_md": str(checklist_md_path),
        "checklist_json": str(checklist_json_path),
        "manifest": str(manifest_path),
    }
    manifest["content_hash"] = sha256_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False))

    checklist_md_path.write_text(render_markdown(include_items, exclusions, questions, manifest), encoding="utf-8")
    write_json(checklist_json_path, {
        "include_items": [asdict(x) for x in include_items],
        "exclusion_items": [asdict(x) for x in exclusions],
        "review_questions": [asdict(x) for x in questions],
        "manifest_summary": manifest,
    })
    write_json(manifest_path, manifest)

    print("[OK] Step 175 patent packet export checklist generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] include_candidate_count   : {manifest['include_candidate_count']}")
    print(f"[OK] default_include_count     : {manifest['default_include_count']}")
    print(f"[OK] exclusion_pattern_count   : {manifest['exclusion_pattern_count']}")
    print(f"[OK] review_question_count     : {manifest['review_question_count']}")
    print(f"[OK] export_zip_created        : {manifest['export_zip_created']}")
    print(f"[OK] attorney_review_required  : {manifest['attorney_review_required']}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 175 patent packet export checklist")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/patent_packet_export_checklist_step175_selftest")
    parser.add_argument("--mode", default="sample", choices=["sample", "template"])
    args = parser.parse_args()
    run(Path(args.output_root), mode=args.mode)


if __name__ == "__main__":
    main()
