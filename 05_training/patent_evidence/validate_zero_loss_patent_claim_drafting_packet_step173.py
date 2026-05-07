from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_FALSE_FLAGS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "train_allowed",
    "legal_novelty_opinion_provided",
    "filing_ready_without_attorney_review",
]

REQUIRED_OUTPUT_KEYS = [
    "packet_md",
    "packet_json",
    "claim_element_matrix",
    "dependent_claim_candidates",
    "evidence_to_claim_mapping",
    "prior_art_differentiation_matrix",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)

    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != "ZERO_LOSS_PATENT_CLAIM_DRAFTING_PACKET_READY_NONCLAIM":
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")
    if not bool(m.get("not_legal_opinion", False)):
        raise RuntimeError("not_legal_opinion must be true")
    if not bool(m.get("attorney_review_required", False)):
        raise RuntimeError("attorney_review_required must be true")

    for key in EXPECTED_FALSE_FLAGS:
        if bool(m.get(key, True)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")

    if int(m.get("pipeline_step_count", 0)) < 13:
        raise RuntimeError("pipeline_step_count must cover Step 160~172")
    if int(m.get("claim_element_count", 0)) < 7:
        raise RuntimeError("claim_element_count too small")
    if int(m.get("dependent_claim_candidate_count", 0)) < 5:
        raise RuntimeError("dependent_claim_candidate_count too small")
    if int(m.get("prior_art_differentiation_count", 0)) < 3:
        raise RuntimeError("prior_art_differentiation_count too small")

    outputs = m.get("outputs", {})
    hashes = m.get("file_sha256", {})
    for key in REQUIRED_OUTPUT_KEYS:
        if key not in outputs:
            raise RuntimeError(f"missing output key: {key}")
        p = Path(outputs[key])
        if not p.exists():
            raise RuntimeError(f"output file missing: {p}")
        if key not in hashes:
            raise RuntimeError(f"missing sha256 for output key: {key}")
        actual = sha256_file(p)
        if actual != hashes[key]:
            raise RuntimeError(f"sha256 mismatch for {key}: {actual} != {hashes[key]}")

    md_text = Path(outputs["packet_md"]).read_text(encoding="utf-8")
    required_terms = [
        "Zero-Loss Pickup Threshold",
        "GATv2",
        "counterfactual ETA",
        "attorney review",
    ]
    for term in required_terms:
        if term not in md_text:
            raise RuntimeError(f"packet markdown missing term: {term}")

    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 173 Zero-Loss patent claim drafting packet")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    m = validate_manifest(Path(args.manifest))
    print("[OK] Step 173 zero-loss patent claim drafting packet validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] audit_status              : {m['audit_status']}")
    print(f"[OK] bundle_status             : {m['bundle_status']}")
    print(f"[OK] claim_element_count       : {m['claim_element_count']}")
    print(f"[OK] dependent_claim_candidate_count : {m['dependent_claim_candidate_count']}")
    print(f"[OK] not_legal_opinion         : {m['not_legal_opinion']}")
    print(f"[OK] attorney_review_required  : {m['attorney_review_required']}")
    print(f"[OK] filing_ready_without_attorney_review : {m['filing_ready_without_attorney_review']}")


if __name__ == "__main__":
    main()
