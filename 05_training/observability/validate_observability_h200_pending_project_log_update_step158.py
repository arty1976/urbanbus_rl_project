from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "OBSERVABILITY_H200_PENDING_PROJECT_LOG_UPDATED_STILL_LOCKED"
EXPECTED_FINAL = "LOCAL_PREPARATION_AND_OBSERVABILITY_HANDOFF_COMPLETE"
EXPECTED_PENDING = "WAITING_FOR_H200_SERVER_AVAILABILITY"
EXPECTED_BLOCK = "H200_SERVER_NOT_AVAILABLE_YET"
EXPECTED_NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"
EXPECTED_CONTROL = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

FALSE_LOCKS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "live_mutation_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "operator_approval_recorded",
    "operator_approval_granted",
]

REQUIRED_SECTION_PHRASES = [
    "Step 158 Observability + H200 Pending Project Log Update",
    "LOCAL_PREPARATION_AND_OBSERVABILITY_HANDOFF_COMPLETE",
    "WAITING_FOR_H200_SERVER_AVAILABILITY",
    "H200_SERVER_NOT_AVAILABLE_YET",
    "actual_execution_allowed = false",
    "train_allowed = false",
    "No local placeholder, mock, expected checklist, or simulated GPU result may be substituted",
    "h200_environment_preflight_result_manifest_step149.py",
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


def validate_manifest(payload: Dict[str, Any]) -> None:
    if payload.get("update_status") != EXPECTED_STATUS:
        fail(f"update_status mismatch: {payload.get('update_status')}")
    if payload.get("final_local_status") != EXPECTED_FINAL:
        fail("final_local_status mismatch")
    if payload.get("h200_pending_status") != EXPECTED_PENDING:
        fail("h200_pending_status mismatch")
    if payload.get("block_reason") != EXPECTED_BLOCK:
        fail("block_reason mismatch")
    if payload.get("next_gate") != EXPECTED_NEXT_GATE:
        fail("next_gate mismatch")
    if payload.get("dashboard_mode") != "MONITORING_ONLY":
        fail("dashboard_mode must be MONITORING_ONLY")
    if payload.get("control_policy") != EXPECTED_CONTROL:
        fail("control_policy mismatch")
    for key in FALSE_LOCKS:
        if bool(payload.get(key)) is not False:
            fail(f"{key} must be false")
    if int(payload.get("hard_failures", -1)) != 0:
        fail("hard_failures must be 0")
    if int(payload.get("component_count", 0)) < 6:
        fail("component_count must include Step 152~157")


def validate_files(payload: Dict[str, Any], project_root: Path) -> None:
    output_files = payload.get("output_files", {})
    section_path = Path(output_files.get("project_log_step158_section", ""))
    preview_path = Path(output_files.get("project_log_step158_updated_preview", ""))
    csv_path = Path(output_files.get("observability_h200_pending_component_status", ""))
    for p in (section_path, preview_path, csv_path):
        if not p.exists():
            fail(f"required output file missing: {p}")
    section = section_path.read_text(encoding="utf-8-sig")
    for phrase in REQUIRED_SECTION_PHRASES:
        if phrase not in section:
            fail(f"section missing required phrase: {phrase}")
    if bool(payload.get("project_log_updated")):
        project_log = Path(payload.get("project_log", project_root / "project_log.md"))
        if not project_log.exists():
            fail("project_log_updated=true but project_log missing")
        log_text = project_log.read_text(encoding="utf-8-sig")
        for phrase in REQUIRED_SECTION_PHRASES[:4]:
            if phrase not in log_text:
                fail(f"project_log missing phrase: {phrase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path}")
    payload = load_json(manifest_path)
    validate_manifest(payload)
    validate_files(payload, project_root)

    print("[OK] Step 158 observability + H200 pending project log update validation PASS")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] status  : {payload.get('update_status')}")
    print(f"[OK] pending : {payload.get('h200_pending_status')}")


if __name__ == "__main__":
    main()
