from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/promotion_candidate/b0c_validation_promotion_candidate_manifest.json",
    )
    args = parser.parse_args()

    path = Path(args.manifest)

    if not path.exists():
        raise SystemExit(f"[FAIL] missing manifest: {path}")

    payload = load_json(path)
    failures = []

    if payload.get("artifact_version") != "b0c_validation_promotion_candidate_manifest_v1":
        failures.append("artifact_version mismatch")

    if payload.get("baseline_id") != "B0C_causal_shadow_v1":
        failures.append("baseline_id must be B0C_causal_shadow_v1")

    if payload.get("condition_id") != "B0C":
        failures.append("condition_id must be B0C")

    if payload.get("promotion_candidate") is not True:
        failures.append("promotion_candidate must be true")

    if payload.get("promotion_released") is not False:
        failures.append("promotion_released must remain false")

    status = payload.get("manifest_status")
    allowed_status = {
        "PROMOTION_CANDIDATE_STILL_LOCKED",
        "PROMOTION_CANDIDATE_STILL_LOCKED_WITH_WARNINGS",
    }
    if status not in allowed_status:
        failures.append(f"manifest_status must be one of {sorted(allowed_status)}, got {status}")

    source = payload.get("validation_source", {})
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        failures.append("validation_source.audit_status must be PASS or PASS_WITH_WARNINGS")

    if int(source.get("hard_failure_count", -1)) != 0:
        failures.append("hard_failure_count must be 0")

    guards = payload.get("claim_guards", {})
    false_keys = [
        "causal_comparison_allowed",
        "paper_level_claim_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "auto_promotion_allowed",
    ]

    for key in false_keys:
        if guards.get(key) is not False:
            failures.append(f"{key} must be false")

    if guards.get("operator_approval_required") is not True:
        failures.append("operator_approval_required must be true")

    if guards.get("explicit_release_gate_required") is not True:
        failures.append("explicit_release_gate_required must be true")

    required_release_gates = payload.get("required_release_gates", [])
    if not required_release_gates:
        failures.append("required_release_gates must not be empty")

    prohibited = payload.get("prohibited_current_use", [])
    if not any("real-world" in str(x) or "real-world operational improvement" in str(x) for x in prohibited):
        failures.append("prohibited_current_use must block real-world improvement claims")

    if failures:
        print("[FAIL] B0C promotion candidate manifest validation failed")
        for item in failures:
            print("[FAIL]", item)
        raise SystemExit(1)

    print("[OK] B0C promotion candidate manifest validation PASS")
    print("[OK] manifest:", path)
    print("[OK] manifest_status:", status)
    print("[OK] promotion_candidate: true")
    print("[OK] promotion_released: false")
    print("[OK] claim_guard: false")


if __name__ == "__main__":
    main()
