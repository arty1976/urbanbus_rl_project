from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_FALSE_FLAGS = [
    "export_zip_created",
    "export_zip_creation_allowed",
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


def validate_manifest(path: Path, require_pass: bool = True) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)

    if m.get("bundle_status") != "ATTORNEY_PACKET_DRY_RUN_EXPORTER_READY_NONCLAIM":
        raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")
    if m.get("dry_run_only") is not True:
        raise RuntimeError("dry_run_only must be true")
    if m.get("attorney_review_required") is not True:
        raise RuntimeError("attorney_review_required must be true")
    for key in EXPECTED_FALSE_FLAGS:
        if m.get(key) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
    if require_pass and m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if require_pass and int(m.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"hard_failures must be 0, got {m.get('hard_failures')}")
    if int(m.get("include_candidate_count", 0)) <= 0:
        raise RuntimeError("include_candidate_count must be positive")

    for label, p in m.get("output_files", {}).items():
        out = Path(p)
        if not out.exists():
            raise RuntimeError(f"output file missing for {label}: {out}")

    print("[OK] Step 176 attorney packet dry-run exporter validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] include_candidate_count   : {m.get('include_candidate_count')}")
    print(f"[OK] included_existing_count   : {m.get('included_existing_count')}")
    print(f"[OK] missing_required_count    : {m.get('missing_required_count')}")
    print(f"[OK] export_zip_created        : {m.get('export_zip_created')}")
    print(f"[OK] filing_ready_without_attorney_review : {m.get('filing_ready_without_attorney_review')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 176 dry-run export manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), require_pass=not bool(args.allow_blocked))


if __name__ == "__main__":
    main()
