from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

LOCKED_FALSE_FLAGS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "train_allowed",
    "winner_selected",
    "actual_evidence_execution_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def validate_manifest(path: Path, require_ready: bool = False, allow_template: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)

    for key in LOCKED_FALSE_FLAGS:
        if bool(m.get(key, False)) is not False:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")

    if not allow_template and bool(m.get("template_only", False)):
        raise RuntimeError("template_only manifest is not acceptable for execution readiness")

    if require_ready:
        if not bool(m.get("ready_for_actual_like_execution", False)):
            raise RuntimeError("ready_for_actual_like_execution must be true")
        if m.get("audit_status") != "PASS":
            raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
        if m.get("hard_failures"):
            raise RuntimeError(f"hard_failures must be empty, got {m.get('hard_failures')}")

    output_files = m.get("output_files", {})
    for label in ("report", "data_quality_report", "command_plan"):
        p = output_files.get(label)
        if not p:
            raise RuntimeError(f"missing output_files.{label}")
        if not Path(p).exists():
            raise RuntimeError(f"output file missing: {label}={p}")

    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 169 readiness manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--allow-template", action="store_true")
    args = parser.parse_args()

    m = validate_manifest(Path(args.manifest), require_ready=bool(args.require_ready), allow_template=bool(args.allow_template))
    print("[OK] Step 169 actual evidence input readiness checklist validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] bundle_status             : {m.get('bundle_status')}")
    print(f"[OK] ready_for_actual_like_execution : {m.get('ready_for_actual_like_execution')}")
    print(f"[OK] hard_failures             : {len(m.get('hard_failures', []))}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")


if __name__ == "__main__":
    main()
