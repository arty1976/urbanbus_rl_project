from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "baseline_reference_manifest_step141_v1"
REQUIRED_BASELINES = {"B0R", "B1", "B2"}

FORBIDDEN_TRUE_KEYS = [
    "causal_comparison_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "best_reward_claim_allowed",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "winner_selected",
    "trainable_reward_promoted",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    return violations


def validate_payload(payload: Dict[str, Any], require_all_ready: bool = False) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")

    if payload.get("step") != 141:
        errors.append("step_must_be_141")

    status = payload.get("audit_status")
    if status not in ("PASS", "PASS_WITH_INCOMPLETE_BASELINES"):
        errors.append("invalid_audit_status")

    refs = payload.get("baseline_references", [])
    baseline_ids = {str(r.get("baseline_id")) for r in refs}
    if baseline_ids != REQUIRED_BASELINES:
        errors.append(f"baseline_ids_mismatch: {sorted(baseline_ids)}")

    if int(payload.get("required_baseline_count", -1)) != 3:
        errors.append("required_baseline_count_must_be_3")

    ready_count = int(payload.get("ready_baseline_count", -1))
    if ready_count < 0 or ready_count > 3:
        errors.append("ready_baseline_count_out_of_range")

    if require_all_ready and ready_count != 3:
        errors.append("not_all_baselines_ready")

    errors.extend(recursive_forbidden_true(payload))

    for ref in refs:
        if ref.get("baseline_id") not in REQUIRED_BASELINES:
            errors.append(f"unexpected_baseline_id: {ref.get('baseline_id')}")
        if bool(ref.get("causal_comparison_allowed", False)):
            errors.append(f"{ref.get('baseline_id')}: causal_comparison_allowed_true")
        if bool(ref.get("paper_level_claim_allowed", False)):
            errors.append(f"{ref.get('baseline_id')}: paper_level_claim_allowed_true")
        if ref.get("reference_status") not in {
            "READY_AS_NONCAUSAL_BASELINE_REFERENCE",
            "PARTIAL_REFERENCE_HAS_WINDOW_KPI",
            "INCOMPLETE_REFERENCE_MISSING_CANONICAL_OUTPUTS",
            "MISSING",
        }:
            errors.append(f"{ref.get('baseline_id')}: invalid_reference_status")

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-all-ready", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_all_ready=args.require_all_ready)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 141 B-group baseline reference manifest validation PASS")


if __name__ == "__main__":
    main()
