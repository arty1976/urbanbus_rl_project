from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ALLOWED_STATUSES = {
    "ACTUAL_EVIDENCE_EXECUTION_RELEASE_CHECKLIST_READY_STILL_LOCKED",
    "ACTUAL_EVIDENCE_EXECUTION_RELEASE_CHECKLIST_RELEASED_FOR_EXECUTION_NONCLAIM",
}


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def validate_manifest(path: Path, expect_locked: bool = False, expect_released: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest does not exist: {path}")

    m = load_json(path)

    if m.get("artifact_version") != "actual_evidence_execution_release_checklist_step171_v1":
        raise RuntimeError(f"unexpected artifact_version: {m.get('artifact_version')}")
    if int(m.get("step", -1)) != 171:
        raise RuntimeError(f"unexpected step: {m.get('step')}")
    if m.get("bundle_status") not in ALLOWED_STATUSES:
        raise RuntimeError(f"unexpected bundle_status: {m.get('bundle_status')}")

    for key in ("paper_level_claim_allowed", "causal_performance_claim_allowed", "actual_operational_claim_allowed", "train_allowed", "winner_selected"):
        if bool(m.get(key, True)):
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected=False")

    if int(m.get("command_count", 0)) < 6:
        raise RuntimeError(f"command_count must be >= 6, got {m.get('command_count')}")

    if "checks" not in m or not isinstance(m["checks"], list) or not m["checks"]:
        raise RuntimeError("checks list missing or empty")

    hard_failures = int(m.get("hard_failures", -1))
    failed_count = int(m.get("failed_check_count", -1))
    if hard_failures != failed_count:
        raise RuntimeError(f"hard_failures/fail count mismatch: {hard_failures} vs {failed_count}")

    allowed = bool(m.get("actual_evidence_execution_allowed", False))
    command_allowed = bool(m.get("command_execution_allowed", False))
    if allowed != command_allowed:
        raise RuntimeError("actual_evidence_execution_allowed and command_execution_allowed must match")

    if expect_locked and allowed:
        raise RuntimeError("expected locked checklist, but execution is allowed")

    if expect_released:
        if not allowed:
            raise RuntimeError("expected released checklist, but execution is not allowed")
        if m.get("audit_status") != "PASS":
            raise RuntimeError("released checklist must have audit_status PASS")
        if not bool(m.get("operator_approval_granted", False)):
            raise RuntimeError("released checklist requires operator_approval_granted=true")
        if not bool(m.get("release_manifest_committed", False)):
            raise RuntimeError("released checklist requires release_manifest_committed=true")

    for label, f in m.get("output_files", {}).items():
        p = Path(f)
        if not p.exists():
            raise RuntimeError(f"output file missing for {label}: {p}")

    print("[OK] Step 171 actual evidence execution release checklist validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] bundle_status             : {m.get('bundle_status')}")
    print(f"[OK] actual_evidence_execution_allowed : {m.get('actual_evidence_execution_allowed')}")
    print(f"[OK] command_execution_allowed : {m.get('command_execution_allowed')}")
    print(f"[OK] hard_failures             : {m.get('hard_failures')}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 171 actual evidence execution release checklist")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expect-locked", action="store_true")
    parser.add_argument("--expect-released", action="store_true")
    args = parser.parse_args()

    validate_manifest(Path(args.manifest), expect_locked=args.expect_locked, expect_released=args.expect_released)


if __name__ == "__main__":
    main()
