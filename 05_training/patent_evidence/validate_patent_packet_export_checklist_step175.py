from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_FALSE_FLAGS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "train_allowed",
    "legal_novelty_opinion_provided",
    "filing_ready_without_attorney_review",
    "export_zip_created",
    "export_zip_creation_allowed",
]


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    m = read_json(manifest_path)
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != "PATENT_PACKET_EXPORT_CHECKLIST_READY_NONCLAIM":
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")
    if not m.get("attorney_review_required", False):
        raise RuntimeError("attorney_review_required must be true")
    if not m.get("not_legal_opinion", False):
        raise RuntimeError("not_legal_opinion must be true")
    for key in EXPECTED_FALSE_FLAGS:
        if bool(m.get(key, False)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
    if int(m.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"hard_failures must be 0, got {m.get('hard_failures')}")

    out = m.get("output_files", {})
    required = ["include_index", "exclusion_index", "open_questions", "checklist_md", "checklist_json", "manifest"]
    for key in required:
        p = Path(out.get(key, ""))
        if not p.exists():
            raise RuntimeError(f"missing output file {key}: {p}")

    includes = read_csv(Path(out["include_index"]))
    exclusions = read_csv(Path(out["exclusion_index"]))
    questions = read_csv(Path(out["open_questions"]))
    if len(includes) != int(m.get("include_candidate_count", -1)):
        raise RuntimeError("include_candidate_count mismatch")
    if len(exclusions) != int(m.get("exclusion_pattern_count", -1)):
        raise RuntimeError("exclusion_pattern_count mismatch")
    if len(questions) != int(m.get("review_question_count", -1)):
        raise RuntimeError("review_question_count mismatch")

    include_paths = "\n".join(x.get("path", "") for x in includes)
    if "*_selftest" not in "\n".join(x.get("pattern", "") for x in exclusions):
        raise RuntimeError("selftest exclusion pattern missing")
    if "secret" not in "\n".join(x.get("pattern", "") for x in exclusions).lower():
        raise RuntimeError("secret exclusion pattern missing")
    if "step173" not in include_paths:
        raise RuntimeError("Step 173 include candidate missing")
    if "step174" not in include_paths:
        raise RuntimeError("Step 174 include candidate missing")

    print("[OK] Step 175 patent packet export checklist validation PASS")
    print(f"[OK] manifest                : {manifest_path}")
    print(f"[OK] include_candidate_count : {m.get('include_candidate_count')}")
    print(f"[OK] default_include_count   : {m.get('default_include_count')}")
    print(f"[OK] exclusion_pattern_count : {m.get('exclusion_pattern_count')}")
    print(f"[OK] review_question_count   : {m.get('review_question_count')}")
    print(f"[OK] export_zip_created      : {m.get('export_zip_created')}")
    print(f"[OK] attorney_review_required: {m.get('attorney_review_required')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 175 patent packet export checklist")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    validate_manifest(Path(args.manifest))


if __name__ == "__main__":
    main()
