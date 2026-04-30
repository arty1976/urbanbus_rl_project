from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "H200_OBSERVABILITY_OPERATOR_RUNBOOK_READY_MONITORING_ONLY_STILL_LOCKED"
EXPECTED_MODE = "MONITORING_ONLY"
EXPECTED_CONTROL = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
REQUIRED_COMPONENT_STEPS = {"152", "153", "154", "155"}
REQUIRED_OUTPUT_KEYS = {
    "manifest",
    "runbook_md",
    "h200_command_cheatsheet_sh",
    "local_tensorboard_tunnel_cheatsheet_ps1",
    "operator_quick_checklist_md",
    "component_status_csv",
}
REQUIRED_H200_COMMAND_NAMES = {
    "view_step155_status_page",
    "tail_jsonl_metrics",
    "run_gpu_probe_once",
    "launch_tensorboard_read_only",
}
FORBIDDEN_COMMAND_SUBSTRINGS = [
    "train_mappo_actual.py",
    "--actual-execution-allowed true",
    "actual_execution_allowed=true",
    "train_allowed=true",
    "reward_weight_live_update",
    "live_mutation_allowed=true",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def require_false(payload: Dict[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        fail(f"{key} must be false, got {payload.get(key)!r}")


def validate_commands(commands: List[Dict[str, Any]]) -> None:
    for item in commands:
        name = str(item.get("name", ""))
        cmd = str(item.get("command", ""))
        if not name:
            fail("command item missing name")
        if not cmd:
            fail(f"command {name} missing command")
        lowered = cmd.lower()
        for forbidden in FORBIDDEN_COMMAND_SUBSTRINGS:
            if forbidden.lower() in lowered:
                fail(f"forbidden command substring in {name}: {forbidden}")


def validate_manifest(path: Path) -> Dict[str, Any]:
    payload = load_json(path)

    if payload.get("runbook_status") != EXPECTED_STATUS:
        fail(f"runbook_status mismatch: {payload.get('runbook_status')}")
    if payload.get("dashboard_mode") != EXPECTED_MODE:
        fail(f"dashboard_mode mismatch: {payload.get('dashboard_mode')}")
    if payload.get("control_policy") != EXPECTED_CONTROL:
        fail(f"control_policy mismatch: {payload.get('control_policy')}")

    for key in (
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "live_mutation_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "operator_approval_recorded",
        "operator_approval_granted",
    ):
        require_false(payload, key)

    if int(payload.get("hard_failures", -1)) != 0:
        fail(f"hard_failures must be 0, got {payload.get('hard_failures')}")

    comp = payload.get("component_status")
    if not isinstance(comp, dict):
        fail("component_status missing")
    rows = comp.get("rows")
    if not isinstance(rows, list):
        fail("component_status.rows missing")
    steps = {str(row.get("step")) for row in rows}
    if steps != REQUIRED_COMPONENT_STEPS:
        fail(f"component steps mismatch: {steps}")

    out = payload.get("output_files")
    if not isinstance(out, dict):
        fail("output_files missing")
    missing_keys = REQUIRED_OUTPUT_KEYS - set(out)
    if missing_keys:
        fail(f"output_files missing keys: {sorted(missing_keys)}")

    for key in REQUIRED_OUTPUT_KEYS:
        out_path = Path(out[key])
        if not out_path.exists():
            fail(f"output file missing: {key} -> {out_path}")

    h200_commands = payload.get("h200_observation_commands")
    local_commands = payload.get("local_observation_commands")
    if not isinstance(h200_commands, list) or not h200_commands:
        fail("h200_observation_commands missing")
    if not isinstance(local_commands, list) or not local_commands:
        fail("local_observation_commands missing")

    command_names = {str(item.get("name")) for item in h200_commands}
    if command_names != REQUIRED_H200_COMMAND_NAMES:
        fail(f"h200 command names mismatch: {command_names}")

    validate_commands(h200_commands)
    validate_commands(local_commands)

    runbook_text = Path(out["runbook_md"]).read_text(encoding="utf-8")
    for needle in (
        EXPECTED_STATUS,
        EXPECTED_MODE,
        EXPECTED_CONTROL,
        "train_allowed: `False`",
        "live_mutation_allowed: `False`",
    ):
        if needle not in runbook_text:
            fail(f"runbook missing required text: {needle}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 156 H200 observability operator runbook manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    payload = validate_manifest(Path(args.manifest))
    print("[OK] Step 156 H200 observability operator runbook validation PASS")
    print(f"[OK] manifest     : {args.manifest}")
    print(f"[OK] components   : {payload['component_status']['present_count']}/{payload['component_status']['expected_count']}")
    print(f"[OK] warnings     : {len(payload.get('warnings', []))}")
    print(f"[OK] runbook      : {payload['output_files']['runbook_md']}")


if __name__ == "__main__":
    main()
