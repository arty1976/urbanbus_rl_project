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
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_manifest(path: Path, strict_file_coverage: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != "FINAL_ZERO_LOSS_PATENT_EVIDENCE_HANDOFF_INDEX_READY_NONCLAIM":
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")
    if not bool(m.get("handoff_index_ready", False)):
        raise RuntimeError("handoff_index_ready must be true")
    if not bool(m.get("template_only", False)):
        raise RuntimeError("template_only must be true")
    for key in EXPECTED_FALSE:
        if bool(m.get(key, True)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")
        if bool(m.get("guards", {}).get(key, True)) is not False:
            raise RuntimeError(f"guards.{key} mismatch: {m.get('guards', {}).get(key)}, expected=False")
    if int(m.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"hard_failures must be 0, got {m.get('hard_failures')}")
    if len(m.get("covered_steps", [])) < 12:
        raise RuntimeError("covered_steps must include Step 160-171")
    if len(m.get("pipeline_chain", [])) < 6:
        raise RuntimeError("pipeline_chain is too short")
    file_scan = m.get("file_scan", {})
    if strict_file_coverage and int(file_scan.get("missing_file_count", 0)) != 0:
        raise RuntimeError(f"strict file coverage requested but missing_file_count={file_scan.get('missing_file_count')}")
    for key, out_path in m.get("output_files", {}).items():
        if not Path(out_path).exists():
            raise RuntimeError(f"output file missing for {key}: {out_path}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 172 final handoff index")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict-file-coverage", action="store_true")
    args = parser.parse_args()
    m = validate_manifest(Path(args.manifest), strict_file_coverage=bool(args.strict_file_coverage))
    print("[OK] Step 172 final zero-loss patent evidence handoff index validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] bundle_status             : {m.get('bundle_status')}")
    print(f"[OK] present_file_count        : {m.get('file_scan', {}).get('present_file_count')}")
    print(f"[OK] missing_file_count        : {m.get('file_scan', {}).get('missing_file_count')}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")


if __name__ == "__main__":
    main()
