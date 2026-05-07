from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


BEGIN_MARKER = "<!-- STEP168_ZERO_LOSS_PATENT_EVIDENCE_LOG_BEGIN -->"
END_MARKER = "<!-- STEP168_ZERO_LOSS_PATENT_EVIDENCE_LOG_END -->"


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read text: {path}")


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    m = load_json(manifest_path)

    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")

    if m.get("log_update_status") not in {"PROJECT_LOG_UPDATED", "DRY_RUN_READY"}:
        raise RuntimeError(f"unexpected log_update_status: {m.get('log_update_status')}")

    if m.get("documented_steps") != [160, 161, 162, 163, 164, 165, 166, 167]:
        raise RuntimeError(f"documented_steps mismatch: {m.get('documented_steps')}")

    false_flags = [
        "actual_route_aware_rollout_evidence_ready",
        "actual_operational_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "train_allowed",
        "performance_claim_allowed",
    ]
    for key in false_flags:
        if bool(m.get(key, True)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")

    project_log = Path(str(m.get("project_log", "")))
    if not project_log.exists():
        raise RuntimeError(f"project_log not found: {project_log}")
    text = read_text_any(project_log)

    required_tokens: List[str] = [
        BEGIN_MARKER,
        END_MARKER,
        "Step 160",
        "Step 161",
        "Step 162",
        "Step 163",
        "Step 164",
        "Step 165",
        "Step 166",
        "Step 167",
        "Step 168",
        "paper_level_claim_allowed = false",
        "causal_performance_claim_allowed = false",
        "actual_operational_claim_allowed = false",
        "actual_route_aware_rollout_evidence_ready = false",
        "Zero-Loss Pickup Threshold",
    ]
    missing = [t for t in required_tokens if t not in text]
    if missing:
        raise RuntimeError(f"project_log missing required tokens: {missing}")

    section_path = Path(str(m.get("section_path", "")))
    if not section_path.exists():
        raise RuntimeError(f"section_path not found: {section_path}")

    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 168 project log update")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    m = validate_manifest(Path(args.manifest))
    print("[OK] Step 168 zero-loss patent evidence project log update validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] log_update_status         : {m.get('log_update_status')}")
    print(f"[OK] project_log               : {m.get('project_log')}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")


if __name__ == "__main__":
    main()
