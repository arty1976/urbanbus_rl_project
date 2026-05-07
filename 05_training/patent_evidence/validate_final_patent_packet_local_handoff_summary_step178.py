from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_FALSE = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "train_allowed",
    "zip_creation_allowed",
    "export_zip_created",
    "filing_ready_without_attorney_review",
    "legal_novelty_opinion_provided",
]

REQUIRED_OUTPUT_KEYS = [
    "summary_md",
    "summary_json",
    "pipeline_step_index_csv",
    "locked_guard_index_csv",
    "manifest_json",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != "FINAL_PATENT_PACKET_LOCAL_HANDOFF_SUMMARY_READY_STILL_LOCKED":
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")
    if int(m.get("pipeline_step_count", -1)) != 19:
        raise RuntimeError(f"pipeline_step_count must be 19 for Step 160~178, got {m.get('pipeline_step_count')}")
    if not bool(m.get("local_preparation_closed", False)):
        raise RuntimeError("local_preparation_closed must be true")
    if not bool(m.get("technical_packet_ready_for_attorney_review", False)):
        raise RuntimeError("technical_packet_ready_for_attorney_review must be true")
    if not bool(m.get("attorney_review_required", False)):
        raise RuntimeError("attorney_review_required must be true")
    if not bool(m.get("not_legal_opinion", False)):
        raise RuntimeError("not_legal_opinion must be true")
    for key in EXPECTED_FALSE:
        if bool(m.get(key, True)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
    if int(m.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"hard_failures must be 0, got {m.get('hard_failures')}")
    output_files = m.get("output_files", {})
    for key in REQUIRED_OUTPUT_KEYS:
        p = Path(output_files.get(key, ""))
        if not p.exists():
            raise RuntimeError(f"missing output file for {key}: {p}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 178 final patent packet local handoff summary")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    m = validate_manifest(Path(args.manifest))
    print("[OK] Step 178 final patent packet local handoff summary validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] audit_status              : {m['audit_status']}")
    print(f"[OK] bundle_status             : {m['bundle_status']}")
    print(f"[OK] pipeline_step_count       : {m['pipeline_step_count']}")
    print(f"[OK] local_preparation_closed  : {m['local_preparation_closed']}")
    print(f"[OK] zip_creation_allowed      : {m['zip_creation_allowed']}")
    print(f"[OK] export_zip_created        : {m['export_zip_created']}")
    print(f"[OK] filing_ready_without_attorney_review : {m['filing_ready_without_attorney_review']}")


if __name__ == "__main__":
    main()
