from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

REQUIRED_FALSE = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "train_allowed",
    "legal_novelty_opinion_provided",
    "filing_ready_without_attorney_review",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != "PATENT_ATTORNEY_REVIEW_PACKET_INDEX_READY_NONCLAIM":
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")
    if int(m.get("pipeline_step_count", 0)) < 14:
        raise RuntimeError("pipeline_step_count must be >= 14")
    if int(m.get("attachment_count", 0)) < 10:
        raise RuntimeError("attachment_count must be >= 10")
    if int(m.get("review_question_count", 0)) < 6:
        raise RuntimeError("review_question_count must be >= 6")
    if not bool(m.get("not_legal_opinion", False)):
        raise RuntimeError("not_legal_opinion must be true")
    if not bool(m.get("attorney_review_required", False)):
        raise RuntimeError("attorney_review_required must be true")
    for key in REQUIRED_FALSE:
        if bool(m.get(key, True)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
    for label, out_path in m.get("outputs", {}).items():
        if label == "manifest":
            continue
        if not Path(out_path).exists():
            raise RuntimeError(f"declared output missing: {label} -> {out_path}")
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    m = validate_manifest(Path(args.manifest))
    print("[OK] Step 174 patent attorney review packet index validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] bundle_status             : {m.get('bundle_status')}")
    print(f"[OK] attachment_count          : {m.get('attachment_count')}")
    print(f"[OK] review_question_count     : {m.get('review_question_count')}")
    print(f"[OK] not_legal_opinion         : {m.get('not_legal_opinion')}")
    print(f"[OK] attorney_review_required  : {m.get('attorney_review_required')}")
    print(f"[OK] filing_ready_without_attorney_review : {m.get('filing_ready_without_attorney_review')}")


if __name__ == "__main__":
    main()
