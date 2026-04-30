from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "OBSERVABILITY_DASHBOARD_INDEX_READY_MONITORING_ONLY_STILL_LOCKED"
EXPECTED_MODE = "MONITORING_ONLY"
EXPECTED_CONTROL = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
REQUIRED_STEPS = {"152", "153", "154"}
LOCK_FIELDS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "live_mutation_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise ValidationError(f"failed to read json: {path}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate(payload: Dict[str, Any]) -> None:
    require(payload.get("index_status") == EXPECTED_STATUS, "index_status mismatch")
    require(payload.get("dashboard_mode") == EXPECTED_MODE, "dashboard_mode mismatch")
    require(payload.get("control_policy") == EXPECTED_CONTROL, "control_policy mismatch")
    require(bool(payload.get("monitoring_only")) is True, "monitoring_only must be true")
    require(bool(payload.get("release_step")) is False, "release_step must be false")

    locks = payload.get("locks")
    require(isinstance(locks, dict), "locks must be a dict")
    for key in LOCK_FIELDS:
        require(key in payload, f"top-level {key} missing")
        require(key in locks, f"locks.{key} missing")
        require(bool(payload[key]) is False, f"top-level {key} must be false")
        require(bool(locks[key]) is False, f"locks.{key} must be false")

    components = payload.get("components")
    require(isinstance(components, list), "components must be a list")
    observed_steps = {str(c.get("step")) for c in components}
    require(observed_steps == REQUIRED_STEPS, f"component steps mismatch: {sorted(observed_steps)}")

    for c in components:
        require("name" in c, "component.name missing")
        require("manifest_relpath" in c, "component.manifest_relpath missing")
        require("manifest_exists" in c, "component.manifest_exists missing")
        require("expected_status" in c, "component.expected_status missing")
        require("status_matches_expected" in c, "component.status_matches_expected missing")
        require("component_ready" in c, "component.component_ready missing")
        require("lock_violation_fields" in c, "component.lock_violation_fields missing")
        require(isinstance(c.get("lock_violation_fields"), list), "lock_violation_fields must be list")
        require(not c.get("lock_violation_fields"), f"component has lock violations: {c}")

    hard_failures = payload.get("hard_failures")
    require(isinstance(hard_failures, list), "hard_failures must be list")
    require(len(hard_failures) == 0, f"hard_failures must be empty: {hard_failures}")

    output_files = payload.get("output_files")
    require(isinstance(output_files, dict), "output_files must be dict")
    for key in ("manifest", "dashboard_index", "status_page", "operator_quick_view", "component_status_csv"):
        require(key in output_files, f"output_files.{key} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 155 H200 observability dashboard index manifest")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = load_json(manifest_path)
    validate(payload)

    print("[OK] Step 155 H200 observability dashboard index/status page validation PASS")
    print(f"[OK] manifest    : {manifest_path}")
    print(f"[OK] components  : {sorted(str(c.get('step')) for c in payload['components'])}")
    print(f"[OK] warnings    : {len(payload.get('warnings', []))}")
    print(f"[OK] ready       : {payload.get('observability_scaffold_ready')}")


if __name__ == "__main__":
    main()
