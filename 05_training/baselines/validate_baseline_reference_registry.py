from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_BASELINES = {
    "B0R_historical_12kpi_compat",
    "B1_noop",
    "B2_rulebased_calibrated",
    "B0C_causal_shadow_v1",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="artifacts/baseline_v1/baseline_reference_registry/baseline_reference_registry.json",
    )
    args = parser.parse_args()

    path = Path(args.registry)

    if not path.exists():
        raise SystemExit(f"[FAIL] missing registry: {path}")

    payload = load_json(path)
    failures = []

    if payload.get("artifact_version") != "baseline_reference_registry_v1":
        failures.append("artifact_version mismatch")

    if payload.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        failures.append("audit_status must be PASS or PASS_WITH_WARNINGS")

    if payload.get("registry_status") != "LOCKED_REFERENCE_SET":
        failures.append("registry_status must be LOCKED_REFERENCE_SET")

    baselines = payload.get("baselines", {})
    got = set(baselines.keys())

    if got != REQUIRED_BASELINES:
        failures.append(f"baseline set mismatch expected={sorted(REQUIRED_BASELINES)} got={sorted(got)}")

    guards = payload.get("global_claim_guards", {})
    for key in [
        "causal_comparison_allowed",
        "paper_level_claim_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "auto_promotion_allowed",
    ]:
        if guards.get(key) is not False:
            failures.append(f"global guard {key} must be false")

    for baseline_id, item in baselines.items():
        overall = item.get("overall_summary", {})
        if overall.get("causal_comparison_allowed") is not False:
            failures.append(f"{baseline_id}: causal_comparison_allowed must be false")

        if not item.get("release_status"):
            failures.append(f"{baseline_id}: release_status missing")

        if not item.get("comparison_scope"):
            failures.append(f"{baseline_id}: comparison_scope missing")

    b2 = baselines.get("B2_rulebased_calibrated", {})
    b2_validation = b2.get("extra_validation") or {}
    if b2_validation.get("audit_status") != "PASS":
        failures.append("B2 extra validation must be PASS")
    if float(b2_validation.get("intervention_rate", 0.0)) <= 0.0:
        failures.append("B2 intervention_rate must be positive")

    b0c = baselines.get("B0C_causal_shadow_v1", {})
    b0c_validation = b0c.get("extra_validation") or {}
    if b0c_validation.get("audit_status") != "PASS":
        failures.append("B0C simulator validation must be PASS")

    b0c_promotion = b0c.get("promotion_manifest") or {}
    if b0c_promotion.get("promotion_candidate") is not True:
        failures.append("B0C promotion_candidate must be true")
    if b0c_promotion.get("promotion_released") is not False:
        failures.append("B0C promotion_released must remain false")

    readiness = payload.get("comparison_readiness", {})
    if readiness.get("baseline_side_ready") is not True:
        failures.append("baseline_side_ready must be true")
    if readiness.get("actual_policy_side_ready") is not False:
        failures.append("actual_policy_side_ready must be false")
    if readiness.get("requires_h200_actual_a_family_results") is not True:
        failures.append("requires_h200_actual_a_family_results must be true")

    if failures:
        print("[FAIL] baseline reference registry validation failed")
        for item in failures:
            print("[FAIL]", item)
        raise SystemExit(1)

    print("[OK] Baseline reference registry validation PASS")
    print("[OK] registry:", path)
    print("[OK] baseline_count:", len(baselines))
    print("[OK] baseline_side_ready: true")
    print("[OK] actual_policy_side_ready: false")
    print("[OK] claim_guard: false")


if __name__ == "__main__":
    main()
