from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_FALSE_FLAGS = [
    "export_zip_created",
    "zip_creation_allowed",
    "export_zip_creation_allowed",
    "operator_approval_recorded",
    "operator_approval_granted",
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

    if m.get("bundle_status") != "ATTORNEY_PACKET_ZIP_EXPORTER_READY_STILL_LOCKED_NONCLAIM":
        raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")
    if m.get("dry_run_only") is not True:
        raise RuntimeError("dry_run_only must be true")
    if m.get("zip_exporter_still_locked") is not True:
        raise RuntimeError("zip_exporter_still_locked must be true")
    if m.get("attorney_review_required") is not True:
        raise RuntimeError("attorney_review_required must be true")
    for key in EXPECTED_FALSE_FLAGS:
        if m.get(key) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
    if require_pass and m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if require_pass and int(m.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"hard_failures must be 0, got {m.get('hard_failures')}")
    if int(m.get("would_include_count", 0)) <= 0:
        raise RuntimeError("would_include_count must be positive")
    planned_zip_path = Path(str(m.get("planned_zip_path", "")))
    if planned_zip_path.exists():
        raise RuntimeError(f"planned ZIP exists even though Step 177 is locked: {planned_zip_path}")
    for label, p in m.get("output_files", {}).items():
        out = Path(p)
        if not out.exists():
            raise RuntimeError(f"output file missing for {label}: {out}")

    print("[OK] Step 177 attorney packet ZIP exporter still locked validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] would_include_count       : {m.get('would_include_count')}")
    print(f"[OK] missing_at_step177_count  : {m.get('missing_at_step177_count')}")
    print(f"[OK] zip_creation_allowed      : {m.get('zip_creation_allowed')}")
    print(f"[OK] export_zip_created        : {m.get('export_zip_created')}")
    print(f"[OK] filing_ready_without_attorney_review : {m.get('filing_ready_without_attorney_review')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 177 locked ZIP exporter manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), require_pass=not bool(args.allow_blocked))


if __name__ == "__main__":
    main()
